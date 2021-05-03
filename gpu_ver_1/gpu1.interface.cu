#include "cuda_runtime.h"
#include <sys/time.h>
#include <vector>
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>
using namespace std;

struct jacobiParameters{
    int r;
    int s;
    double aRS;
};

__global__ void vector_add(double * A, double * B, double * C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N) C[idx] = A[idx] + B[idx];
}
__device__ double atomicAddF(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
 
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
 
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
 
    return __longlong_as_double(old);
}

__device__ double atomicMaxF(double* address, double val)
{
    unsigned long long int* address_as_i = (unsigned long long int*) address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void matrixMeanMax(double *A, int NRows, int NCols, double *max, double *mean) {
    int xidx = blockIdx.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(xidx<NCols && yidx<NRows){
        
        __shared__ double sum;
        __shared__ double local_max;
        double val = A[xidx + yidx * NCols];

        if (threadIdx.y == 0) {
            sum = 0;
            local_max = 0;
        }
        __syncthreads();

        atomicAddF(&sum, val);
        atomicMaxF(&local_max, fabs(val));
        __syncthreads();

        if(threadIdx.y == 0){
            atomicAddF(&mean[xidx], sum/NRows);
            atomicMaxF(&max[xidx],local_max);
        }
    }
}

__global__ void matrixSelfMulNaive(double *A, double *C , int NRows, int NCols) {
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (xidx < NCols && yidx < NCols){
        double partialSum = 0;
        for (int i=0;i<NRows; i++){
            partialSum += A[yidx + i * NCols] * A[xidx + i * NCols];
        }
        C[xidx + yidx * NCols] = partialSum;
    }
}


__global__ void matrix_multiply2(double *a, double *b, double *ab, unsigned int width)
{
  // calculate the row & column index of the element
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  if (row >= width || col >= width) return;
  float result = 0;


  // do dot product between row of a and column of b
  for(int k = 0; k < width; ++k)
  {
    result += a[row*width+k] * b[k*width+col];
  }

  // write out this thread's result
  ab[row*width+col] = result;
}

#define TILE_WIDTH 32
__global__ void matrix_multiply(double *a, double *b, double *ab, int width)
{
  // create shorthand names for threadIdx & blockIdx
  int tx = threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x,  by = blockIdx.y;

  // calculate the row & column index of the element
  int row = by*blockDim.y + ty;
  int col = bx*blockDim.x + tx;

  double result = 0;

 

  // allocate 2D tiles in __shared__ memory
  __shared__ double s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ double s_b[TILE_WIDTH][TILE_WIDTH];

  // loop over the tiles of the input in phases
  for(int p = 0; p < (int)((width-1)/(TILE_WIDTH))+1; ++p)
  {
    
    // collaboratively load tiles into __shared__
    if ((p*TILE_WIDTH + tx) < width && ((p*TILE_WIDTH + ty) < width)){
        s_a[ty][tx] = a[row*width + (p*TILE_WIDTH + tx)];
        s_b[ty][tx] = b[(p*TILE_WIDTH + ty)*width + col];
    }
    if (row >= width || col >= width) continue;
    // wait until all data is loaded before allowing
    // any thread in this block to continue
    __syncthreads();

    // do dot product between row of s_a and column of s_b
    for(int k = 0; (k < ((TILE_WIDTH > width) ? width : TILE_WIDTH)) && ((p*TILE_WIDTH+k) < width); ++k)
    {
      result += s_a[ty][k] * s_b[k][tx];
    }

    // wait until all threads are finished with the data
    // before allowing any thread in this block to continue
    __syncthreads();
  }
  if (row >= width || col >= width) return;
  // write out this thread's result
  ab[row*width+col] = result;
}

__global__ void matrixNormalize(double * A, double * R, int NRows, int NCols, double *max, double *mean) {
    int xidx = blockIdx.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(xidx<NCols && yidx<NRows){
        R[xidx + yidx * NCols] = (A[xidx + yidx * NCols] - mean[xidx]) / max[xidx];
    }
}

__global__ void eigenvalueEstimate1(double * partialResult, double * A, double * v, unsigned int dim) {
    /*
    Input:
        dim = scalar value = dimension of the problem
        A = matrix (dimension dim x dim)
        v = vector (dimension dim)
    Output:
        partialResult = vector (dimension dim)

    Performs partialResult = A*v

    The kernel is supposed to be called with    grid = dim3(1 + (dim - 1) / (dim < 1024 ? dim : 1024), dim);
                                                block = dim3(dim < 1024 ? dim : 1024, 1);
                                                a third argument = blockDim.x
    */
    int tx = threadIdx.x;
    int bx = blockIdx.x, row = blockIdx.y;
    int bdx = blockDim.x;
    int col = tx + bx * bdx;

    __shared__ double partialSum;

    if(col >= dim) return;

    if(tx == 0) partialSum = 0;

    __syncthreads();

    double val = v[col] * A[col + row * dim];

    atomicAddF(&partialSum, val);
    
    __syncthreads();

    if(tx == 0) atomicAddF(&partialResult[row], partialSum);
}

__global__ void eigenvalueEstimate2(double * result, double * partialResult, double * v, unsigned int dim) {
    /*
    Input:
        dim = scalar value = dimension of the problem
        partialResult = vector (dimension dim)
        v = vector (dimension dim)
    Output:
        result = scalar value

    Performs partialResult = v'*partialResult

    The kernel is supposed to be called with    grid = dim3(1+(dim-1)/k);
                                                block = dim3(k); with k <= 1024
    */

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    __shared__ double partialSum;
    if(tx == 0) partialSum = 0;
    __syncthreads();
    
    int idx = tx + bx * bdx;
    if(idx >= dim) return;
    double val = partialResult[idx]*v[idx];
    atomicAddF(&partialSum, val);
    __syncthreads();
    
    if(tx==0) atomicAddF(result, partialSum);
}

__global__ void vectorOrder(double *in, unsigned int dim) {
    int tx = threadIdx.x;
    int bd = blockDim.x;

    __shared__ unsigned int sorted_flag;
    int n_values = (int) ((dim-1)/(bd)) +1;
    if (n_values % 2 == 1) n_values++;
    int start = tx*n_values;
    if (start >= dim) return;
    int end = start + n_values;
    if(end-1>=dim) end = dim;

    do{
        //fase di swap intra-thread
        for(int i = start; i<end-1; i++){
            double prev = in[i];
            double next = in[i+1];
            if(next>prev){
                in[i] = next;
                in[i+1] = prev;
            }
        }
        __syncthreads();
        //fase swap inter-thread
        if(tx!=0){
            double prev = in[start-1];
            double next = in[start];
            if(next>prev){
                in[start-1] = next;
                in[start] = prev;
            }
        }
        
        if (tx == 0) sorted_flag = 0;
        __syncthreads();

        //fase di check
        for(int i = ((start == 0) ? (start + 1) : start); i<end; i++){
            double prev = in[i-1];
            double next = in[i];
            if(prev>next) continue;
            atomicAdd(&sorted_flag, 1);
        }
        __syncthreads();
        
    } while(sorted_flag>0);
}

__global__ void vectorSum(double *input, double *result, unsigned int width) {
    
    int tx=threadIdx.x;
    int bd= blockDim.x;

    if (tx >= width) return;

    __shared__ double sum;

    double localSum = 0;
	if(tx==0) sum=0;

    for(int i=tx; i<width; i+= bd) localSum += input[i];

    __syncthreads();
        
    atomicAddF(&sum,localSum);

    __syncthreads();
    
    if (tx == 0) *result = sum;
}
/*--------------------------------------------------------------------------------*/

__global__ void jacobiFindRS(double *A, int NRows, jacobiParameters* var) {
    /*
    Input:
        A = input matrix (dimension NRows x NRows)
        NRows = scalar value = number of rows (and columns) of A
        var->aRS = scalar value = max element of the upper, extra-diagonal triangle of A
    Output:
        var->r, var->s = integer values = indexes of the element with value var->aRS

    Find the indexes of the maximum value in the upper, extra-diagonal triangle of A.

    The kernel is supposed to be called with    grid = dim3((int)((nCols-1)/threadsSize)+1, (int)((nCols-1)/threadsSize)+1)

                                                block = dim3(threadsSize,threadsSize);
                                                Where 1 <= threadsSize <= 32 (the greater, the better)
    */

    int xidx =  threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    
    // check if out of bound of the matrix of if in lower triangle
    if (xidx >= NRows || yidx>=xidx ) return;
    
    // each thread check one element of the matrix
    double val = A[xidx + yidx * NRows];
    double max = var->aRS;
    // compare the relative difference between the element of the matrix and the maximum value
    double compareV = (max - (val<0?-val:val))/max ;

    // just for debugging
    // result[xidx + yidx*NRows] = compareV;
    
    // check if the relative difference is low enough
    if ( compareV < 1e-7){
        var->r = yidx;
        var->s = xidx;
        var->aRS = val;
    }

}

__global__ void jacobiFindMax(double *A, int NRows, jacobiParameters *var) {
    /*
    Input:
        A = input matrix (dimension NRows x NRows)
        NRows = scalar value = number of rows (and columns) of A
    Output:
        var->aRS = scalar value = max element of the upper, extra-diagonal triangle of A

    Find the maximum (absolute) value in the upper, extra-diagonal triangle of A.

    The kernel is supposed to be called with    grid = dim3((int)((nCols-1)/threadsSize)+1, (int)((nCols-1)/threadsSize)+1);
                                                block = dim3(threadsSize,threadsSize);
                                                Where 1 <= threadsSize <= 32 (the greater, the better)
    */

    int xidx =  threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    
    //each block finds the local maximum, then updates the global maximum
    __shared__ double local_max;

    // check if out of bound of the matrix or if in lower triangle
    if (xidx >= NRows || yidx>=xidx) return;

    // each thread check one element of the matrix
    double val = A[xidx + yidx * NRows];

    // compute the absolute value of the element
    val = val < 0 ? -val : val;

    // the thread (0,0) of each block takes care of initializing the local maximum
    if (threadIdx.y == 0 && (threadIdx.x == ((blockIdx.x==blockIdx.y) ? 1 : 0))) {
        local_max = 0;
    }

    // synchronize after initializing the local maximum
    __syncthreads();

    // atomically update the local maximum
    atomicMaxF(&local_max, val);

    //synchronize to be sure that each thread updated the local maximum
    __syncthreads();

    // thread (0,0) takes care of updating the global maximum
    if(threadIdx.y == 0 && (threadIdx.x == ((blockIdx.x==blockIdx.y) ? 1 : 0))){
        atomicMaxF(&(var->aRS), local_max);
    }
}

__global__ void jacobiIteration(double *A, int NRows, jacobiParameters *var){
    /*
    Input:
        A = input matrix (dimension NRows x NRows)
        NRows = scalar value = number of rows (and columns) of A
        var = informations about the maximum value to use in the Jacobi iteration
    Output:
        A = matrix = inplace updated

    Perform the Jacobi iteration:
    The operation performed is equivalent to compute Grs'*A*Grs, where Grs is a rotation matrix around the axis r and s of the input space.

    The kernel is supposed to be called with    grid = dim3(1);
                                                block = dim3(nCols);
    */

    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // check if out of bound of the matrix
    if(xidx >= NRows) return;

    // all the threads are in the same block, so the common values are shared
    __shared__ int shared_r;
    __shared__ int shared_s;
    __shared__ double cosFi;
    __shared__ double sinFi;

    // these values will be useful to comute the elements in positions (r,r) and (s,s)
    __shared__ double Arr;
    __shared__ double Ass;
    __shared__ double Ars;

    if (xidx == 0){
        // load the indexes from global memory
        shared_r = var->r;
        shared_s = var->s;

        // load the values needed for elements of row/column r and s
        Arr = A[shared_r + shared_r * NRows];
        Ass = A[shared_s + shared_s * NRows];
        Ars = A[shared_s + shared_r * NRows];

        // compute the components of the rotation block
        double m = (A[var->r + var->r * NRows] - A[var->s + var->s * NRows]) / (2* var->aRS); 
        double t = -m + ((m>=0) ? sqrt(1 + m*m) : (- sqrt(1 + m*m)));    
        cosFi = 1/(sqrt(1 + t*t)); 
        sinFi = t * cosFi;
    }
    
    // synchronize to be sure that thread 0 computed cosFi and sinFi
    __syncthreads();
    
    // load the values that will be updated
    double Air = A[shared_r + xidx * NRows];
    double Ais = A[shared_s + xidx * NRows];
    
    // compute the new values
    double AisNew = Ais * cosFi - Air * sinFi;
    double AirNew = Air * cosFi + Ais * sinFi;
    
    // update
    A[shared_r + xidx * NRows] = AirNew;
    A[shared_s + xidx * NRows] = AisNew;

    // since the resulting matrix will be symmetric, also update the lower triangle
    A[xidx + shared_r * NRows] = AirNew;
    A[xidx + shared_s * NRows] = AisNew;
    
    __syncthreads();
    
    // thread 0 takes care of updating the elements with row or column equal to r or s
    if(xidx == 0){
        // use their previous value to compute the new values
        double ArrNew = Arr*cosFi*cosFi + 2*Ars*cosFi*sinFi + Ass*sinFi*sinFi;
        double AssNew = Ass*cosFi*cosFi - 2*Ars*cosFi*sinFi + Arr*sinFi*sinFi;
        
        // update the values
        A[shared_r + shared_r * NRows] = ArrNew;
        A[shared_s + shared_s * NRows] = AssNew;

        // if the computation is correct, the extra-diagonal elements are forced to 0, so it can be avoided to compute them
        A[shared_s + shared_r * NRows] = 0;
        A[shared_r + shared_s * NRows] = 0;
    }
}

__global__ void matrixMCalculation(double *Mi, double eigenValue, int NCols){
    /*
    Input:
        Mi = input matrix (dimension NRows x NRows)
        NCols = scalar value = number of columns (and rows) of A
    Output:
        Mi = matrix = inplace updated

    Perform Mi = A - eigenvalue * I, where I is the identity matrix.

    The kernel is supposed to be called with    grid = dim3((int)((nCols-1)/threadsSize)+1, (int)((nCols-1)/threadsSize)+1);
                                                block = dim3(threadsSize,threadsSize);
                                                Where 1 <= threadsSize <= 32 (the greater, the better)

    Matrix Mi is supposed to be initially equals to A. Consider executing the kernel inplace.
    */

    int xidx = threadIdx.x + blockIdx.x * blockDim.x;

    // check if out of bound of the matrix or if out of the diagonal
    if (xidx >= NCols) return;

    // each diagonal thread updates one element of the matrix
    Mi[xidx + xidx*NCols] = Mi[xidx + xidx*NCols] - eigenValue;
}

__global__ void transpose( double *I, double *O, unsigned int dim){
    /*
    Input:
        I = input matrix (dimension dim x dim)
    Output:
        dim = scalar value = dimension of the matrix I
        O = output matrix (siensino dim x dim) = the transpose of the input matrix 

    Perform the transposition of I:
    O = I'

    The kernel is supposed to be called with    grid = dim3((int)((nCols-1)/threadsSize)+1, (int)((nCols-1)/threadsSize)+1);
                                                block = dim3(threadsSize,threadsSize);
                                                Where 1 <= threadsSize <= 32 (the greater, the better)

    Consider executing this operation inplace.
    */

    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    // check if out of bound of the matrix
    if (xidx >= dim || yidx >= dim) return;

    // each thread computes one term of the output matrix
    O[xidx + yidx*dim] = I[yidx+xidx*dim];
}

__global__ void choleskyMatrixCalculation( double *I, double *O, unsigned int dim){
    /*
    Input:
        I = input matrix (dimension dim x dim)
    Output:
        dim = scalar value = dimension of the matrix I
        O = output matrix (dimension dim x dim) = upper triangular, cholesky decomposition of the input matrix 

    Performs the Cholesky decomposition of the input matrix

    The kernel is supposed to be called with    grid = dim3(1);
                                                block = dim3(dim < 1024 ? dim : 1024);
    Consider executing this operation inplace.
    */

    int bd = blockDim.x;
    int tx = threadIdx.x;

    // initialize a variable for performing the sums
    double partialSum;

    // compute the position of the biggest diagonal element associated with each thread
    int height = ((int)((dim-1)/bd)) * bd + tx;

    // computation is performed per-row, from the first to the last
    // loop over the rows (from the first row, until the diagonal)
    for(int row=0;row<=height; row++){
        // loop over the columns associated with the current thread
        for(int col=tx; col<dim; col+=bd){

            // apply the formula for the cholesky decomposition
            // start with the element of the input matrix
            partialSum = I[col + row*dim];
            
            // first of all, the element on the diagonal has to be computed
            if (row==col){
                for(int k=0;k<=row-1;k++){
                    partialSum -= O[col + k*dim] * O[col + k*dim];
                }
                O[col + row*dim] = sqrt(partialSum);
            }

            // the computation of the extra-diagonal elements can begin only after the diagonal element has been computed
            __syncthreads();

            // compute the extra-diagonal elements
            if(row!=col){
                for(int k=0;k<=row-1;k++) partialSum -= O[row + k*dim] * O[ col + k*dim];
                O[col + row*dim] = partialSum / O[row + row*dim];
            }
            __syncthreads();             
        }
        __syncthreads();
    }

    // set to 0 all the terms in the lower, extra-diagonal triangle
    for (int col=tx; col<dim; col+=bd){
        for(int i = col+1; i<dim; i++) O[i * dim + col] = 0;
    }
}

__global__ void inverseCholesky(double * L, double * R, unsigned int dim){
    /*
    Input:
        L = input lo matrix (dimension dim x dim)
    Output:
        dim = scalar value = dimension of the matrix L
        R = output matrix (dimension dim x dim) = upper triangular, cholesky decomposition of the input matrix 

    Take in input an upper triangular matrix L and returns its inverse in R.

    The kernel is supposed to be called with    grid = dim3(1);
                                                block = dim3(dim < 1024 ? dim : 1024);
    Consider executing this operation inplace.
    */

    int bd = blockDim.x;
    int tx = threadIdx.x;
    
    // the computation is performed per-diagonals, starting from the main diagonal, moving on, with the sovra-diagonal terms and so on.
    // first of all compute the terms of the main diagonal
    for(int i=tx; i<dim; i+=bd){
        R[i + i * dim] = 1 / L[i + i * dim];
    }

    // synchronize to be sure that the diagonal terms of the output matrix have been computed
    __syncthreads();

    // loop over the diagonals of the upper triangle
    for(int diag=0; diag<dim; diag++){
        //loop over the elements associated with the current thread (if dim > 1024)
        for(int row=tx; row<dim; row+=bd){
            int col=row+diag+1;
            if(col>=dim) break;
            double sum=0;
            for(int j=row+1; j<=col; j++) sum-=L[j+row*dim]*R[j*dim+col];
            R[row*dim+col]=R[row*dim+row]*sum;
        }
        __syncthreads();
    }
    for(int i = 0; i<tx; i++) R[tx * dim + i] = 0;
}

/*--------------------------------OK------------------------------------------------*/
__global__ void normalizeVectorNEW(double * v,unsigned int dim) {
    /*
    Input:
        dim = scalar value = dimension of the problem
        v = vector (dimension dim)
    Output:
        None - in place

    Compute the norm of a given vector

    The kernel is supposed to be called with    grid = dim3(1+(dim-1)/k);
                                                block = dim3(k); with k <= 1024
    */
    int tx = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ double norm; 
    double partialSum =0;

    if (tx == 0) norm =0;
    __syncthreads();

    for(int xidx=tx; xidx<dim; xidx+=bdim){
        double val = v[xidx];
        partialSum += (val * val);
    }
    atomicAddF(&norm, partialSum);
    __syncthreads();

    if (tx == 0) norm = sqrt(norm);
    __syncthreads();

    for(int xidx=tx; xidx<dim; xidx+=bdim){
        v[xidx] = v[xidx] / norm;
    }
}

__global__ void eigenvalueEstimate1NEW(double * partialResult, double * A, double * v, unsigned int dim) {
    /*
    Input:
        dim = scalar value = dimension of the problem
        A = matrix (dimension dim x dim)
        v = vector (dimension dim)
    Output:
        partialResult = vector (dimension dim)

    Performs partialResult = A*v

    The kernel is supposed to be called with    grid = dim3(1 + (dim - 1) / (dim < 1024 ? dim : 1024), dim);
                                                block = dim3(dim < 1024 ? dim : 1024, 1);
    */
    int tx = threadIdx.x;
    int bx = blockIdx.x, row = blockIdx.y;
    int bdx = blockDim.x;
    int col = tx + bx * bdx;

    __shared__ double partialSum;

    if(col >= dim) return;

    if(tx == 0) partialSum = 0;

    __syncthreads();

    double val = v[col] * A[col + row * dim];

    atomicAddF(&partialSum, val);
    
    __syncthreads();

    if(tx == 0) atomicAddF(&partialResult[row], partialSum);
}

__global__ void eigenvalueEstimate2NEW(double * result, double * partialResult, double * v, unsigned int dim) {
    /*
    Input:
        dim = scalar value = dimension of the problem
        partialResult = vector (dimension dim)
        v = vector (dimension dim)
    Output:
        result = scalar value

    Performs partialResult = v'*partialResult

    The kernel is supposed to be called with    grid = dim3(1+(dim-1)/k);
                                                block = dim3(k); with k <= 1024
    */

    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    __shared__ double partialSum;
    if(tx == 0) partialSum = 0;
    __syncthreads();
    
    int idx = tx + bx * bdx;
    if(idx >= dim) return;
    double val = partialResult[idx]*v[idx];
    atomicAddF(&partialSum, val);
    __syncthreads();
    
    if(tx==0) atomicAddF(result, partialSum);
}


__global__ void vectorOrderNEW(double *in, unsigned int dim) {
    int tx = threadIdx.x;
    int bd = blockDim.x;

    __shared__ bool sorted_flag;
    int n_values = (int) ((dim-1)/(bd)) +1;
    if (n_values % 2 == 1) n_values++;
    int start = tx*n_values;
    if (start >= dim) return;
    int end = start + n_values;
    if(end-1>=dim) end = dim;

    do{
        //fase di swap intra-thread
        for(int i = start; i<end-1; i++){
            double prev = in[i];
            double next = in[i+1];
            if(next>prev){
                in[i] = next;
                in[i+1] = prev;
            }
        }
        __syncthreads();
        //fase swap inter-thread
        if(tx!=0){
            double prev = in[start-1];
            double next = in[start];
            if(next>prev){
                in[start-1] = next;
                in[start] = prev;
            }
        }
        
        if (tx == 0) sorted_flag = true;
        __syncthreads();

        //fase di check
        for(int i = ((start == 0) ? (start + 1) : start); i<end; i++){
            double prev = in[i-1];
            double next = in[i];
            if(prev>next) continue;
            sorted_flag = false;
        }
        __syncthreads();
        
    } while(!sorted_flag);
}

//vectorSelectPrincipalComponent<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_EigenValues,d_MaxPrincipalComponent,nCols);

__global__ void vectorSumNEW(double *input, double *result, unsigned int width) {
    
    int tx=threadIdx.x;
    int bd= blockDim.x;

    if (tx >= width) return;

    __shared__ double sum;

    double localSum = 0;
	if(tx==0) sum=0;

    for(int i=tx; i<width; i+= bd) localSum += input[i];

    __syncthreads();
        
    atomicAddF(&sum,localSum);

    __syncthreads();
    
    if (tx == 0) *result = sum;
}

class Interface{
    public:
        
        void matrixNormalizationHost(double * I, double * O, int nRows, int nCols, unsigned int threadsSize=1024){
            /*
                Funzione che normalizza una data matrice di ingresso e ritorna la matrice normalizzata
                I: 
                    double* I : matrice di input
                    double* O : matrice di output 
                    int nRows : numero di righe matrice
                    int nCols : numero di colonne matrice
                    int threadsSize (default 1024) : numero di thread in esecuzione parallela su gpu
                O:
                    //
            */
            
            double *d_I;
            double *d_O;
            double *max = new double[nCols];
            double *mean = new double[nCols];

            cudaMalloc((void**)&d_I, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_O, sizeof(double)*nCols*nRows) ;
            cudaMalloc((void**)&max, sizeof(double)*nCols) ;
            cudaMalloc((void**)&mean, sizeof(double)*nCols) ;
            
            cudaMemcpy(d_I, I, sizeof(double)*nCols*nRows, cudaMemcpyHostToDevice) ;
            cudaMemset(max, 0, sizeof(double)*nCols) ;
            cudaMemset(mean, 0, sizeof(double)*nCols) ;
         
            matrixMeanMax<<<dim3(nCols, (int)(nRows/threadsSize+1)), dim3(1,threadsSize)>>>(d_I, nRows, nCols, max, mean);

            matrixNormalize<<<dim3(nCols, (int)(nRows/threadsSize+1)), dim3(1,threadsSize)>>>(d_I, d_O, nRows, nCols, max, mean);
         
            cudaMemcpy(O, d_O, sizeof(double)*nCols*nRows, cudaMemcpyDeviceToHost);
         
            cudaFree(d_I);
            cudaFree(d_O);
            cudaFree(max);
            cudaFree(mean);
        }

        void matrixSelfMultiplication(double * I, double * O, int nRows, int nCols,unsigned int threadsSize=32, int operationCode=0){
            /*
                Funzione che effettua prodotto della matrice per se stessa
                I: 
                    double* I : matrice di input
                    double* O : matrice di output 
                    int nRows : numero di righe matrice
                    int nCols : numero di colonne matrice
                    int threadsSize (default 1024) : numero di thread in esecuzione parallela su gpu
                O:
                    //
            */

            double *d_I;
            double *d_O;

            cudaMalloc((void**)&d_I, sizeof(double)*nCols*nRows);

            cudaMalloc((void**)&d_O, sizeof(double)*nCols*nCols);

            cudaMemcpy(d_I, I, sizeof(double)*nCols*nRows, cudaMemcpyHostToDevice);

            switch(operationCode){
                case 0:
                    matrixSelfMulNaive<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_I, d_O, nRows, nCols);
                    break;
                default:
                    matrixSelfMulNaive<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1), dim3(threadsSize,threadsSize)>>>(d_I,d_O, nRows, nCols);
                    break;
            }
            
            cudaMemcpy(O, d_O, sizeof(double)*nCols*nCols, cudaMemcpyDeviceToHost);
            
            cudaFree(d_O);
            cudaFree(d_I);
            
            
        }
    
        int jbEigenvaluesFinder(double * I, double * O, int nRows, int nCols,unsigned int threadsSize=32, double threshold=1e-3){
            /*
                Funzione che effettua prodotto della matrice per se stessa
                I: 
                    double* I : matrice di input
                    double* O : matrice di output 
                    int nRows : numer- co di righe matrice
                    int nCols : numero di colonne matrice
                    int threadsSize (default 1024) : numero di thread in esecuzione parallela su gpu
                    double threshold (default 1*10-3) : criterio di stop
                O:
                    //
            */
            // jacobiParameters *hostJParameters;
            double percentuleSum = 0;
            double* tempMatrix = new double[nCols*nCols];
            double* h_EigenValues = new double[nCols];
            double hARS;
            int a=0;
            double result;
            double *d_I;
            double *d_EigenValues;
            double *d_result;
            jacobiParameters *jParameters;
            

            cudaMalloc((void**)&d_I, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&jParameters, sizeof(jacobiParameters));
            cudaMalloc((void**)&d_EigenValues, sizeof(double)*nCols);
            cudaMalloc((void**)&d_result, sizeof(double));
            
           
            cudaMemset(d_EigenValues, 0, sizeof(double)*nCols);
            cudaMemset(jParameters, 0, sizeof(jacobiParameters));
            cudaMemset(d_result, 0, sizeof(double));
            cudaMemcpy(d_I, I, sizeof(double)*nCols*nRows, cudaMemcpyHostToDevice);
            
            jacobiFindMax<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_I, nRows, jParameters);

            jacobiFindRS<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_I, nRows, jParameters);
            
            cout << "Start Eigenvalues iteration.." << endl;
            do{
                jacobiIteration<<<dim3(1),dim3(nCols)>>>(d_I, nRows, jParameters);

                cudaMemset(jParameters, 0, sizeof(jacobiParameters));
                jacobiFindMax<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_I, nRows, jParameters);

                jacobiFindRS<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_I, nRows, jParameters);

                cudaMemcpy(&hARS, &jParameters->aRS, sizeof(double), cudaMemcpyDeviceToHost);
                a++;
                if (a % 4000 == 0) cout << "Iteration Counter: "<<a<<", Max of extradiagonal: "<<hARS<<endl; 
            }while((hARS > 0 ? hARS : -hARS) > threshold);
            cout << "Stop criterion reached after "<< a <<" iterations" << endl;
            cout << "Start copying eigenvalues... ";
	    cudaMemcpy(tempMatrix, d_I, sizeof(double)*nCols*nCols, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
	    cout << "DONE" << endl;

            for(int i=0; i<nCols; i++){
                h_EigenValues[i] = tempMatrix[i + i*nCols] ;        
            }
            cudaMemcpy(d_EigenValues, h_EigenValues, sizeof(double)*nCols, cudaMemcpyHostToDevice);
            
	    cout << "Start sorting eivenvalues... ";
	    vectorOrder<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_EigenValues,nCols);
            cudaDeviceSynchronize();
	    cout << "DONE" << endl;

	    cout << "Start computing the sum... ";
	    vectorSum<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_EigenValues,d_result,nCols);
            cudaDeviceSynchronize();
	    cout << "DONE" << endl;

	    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
            

            cudaMemcpy(O, d_EigenValues, sizeof(double)*nCols, cudaMemcpyDeviceToHost);
            
            
            for (a=0;a<nCols;a++){
                percentuleSum += (O[a]) / result;
                if (percentuleSum > 0.999) break;
            }
            cout << "The 99% information is stored in the first: "<< a <<" components." << endl;
            cout << "The others will be zeroed. :)" << endl;
            

            delete[] tempMatrix;
            delete[] h_EigenValues;

            cudaFree(d_EigenValues);
            cudaFree(d_I);
            cudaFree(jParameters);
            cudaFree(d_result);
            

            return a;
        }

        void eigenvectorsFinder(double * I, double * O, double * lambdaVector, int nRows, int nCols,unsigned int nEigenValues,unsigned int threadsSize=32, double threshold=1e-5){
            /*
                Funzione che effettua prodotto della matrice per se stessa
                I: 
                    double* I : matrice di input
                    double* O : matrice di output 
                    double* lambdaVector : vettore degli autovalori
                    int nRows : numero di righe matrice
                    int nCols : numero di colonne matrice
                    int threadsSize (default 1024) : numero di thread in esecuzione parallela su gpu
                    double threshold (default 1*10-3) : criterio di stop
                O:
                    //
            */
            
            int a;
            // --- Define Host Variables
            double *h_Mi = new double[nCols*nCols];
            double *hX = new double[nCols];
            double *hV0 = new double[nCols];
            double *hVk = new double[nCols];
            double hApproximatedLambda = 0;
            double hApproximatedLambda_pre = 0;
            double *h_Vk_t_A = new double[nCols];
            
            // --- Define Device Variables
            double *d_eigenValues;
            double *d_I;
            double *d_Vk;
            double *d_Mi;
            double *d_Mi_SelfMultiplied;
            double *d_Mi_Inverse;
            double *d_Mi_Cholesky;
            double *d_Mi_CholeskyInverse;
            double *d_Mi_CholeskyInverse_Transpose;
            double *d_Mi_MoorePenrose;
            double *d_Mi_Transient;
            double *partialResultLambdaApproximation;
            double *d_X;
            double *d_LambdaEstimated;
            

                    
            // --- Istanciate memory on Device
            cudaMalloc((void**)&d_LambdaEstimated, sizeof(double));
            cudaMalloc((void**)&d_X, nCols*sizeof(double));
            cudaMalloc((void**)&d_Vk, sizeof(double)*nCols);
            cudaMalloc((void**)&d_Mi, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_Mi_MoorePenrose, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_Mi_SelfMultiplied, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_Mi_Inverse, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_Mi_Cholesky, sizeof(double)*nRows*nRows);
            cudaMalloc((void**)&d_Mi_CholeskyInverse, sizeof(double)*nRows*nRows);
            cudaMalloc((void**)&d_Mi_CholeskyInverse_Transpose, sizeof(double)*nRows*nRows);
            cudaMalloc((void**)&d_Mi_Transient, sizeof(double)*nRows*nRows);
            cudaMalloc((void**)&d_I, sizeof(double)*nCols*nRows);
            cudaMalloc((void**)&d_eigenValues, sizeof(double) * nCols);
            cudaMalloc((void**)&partialResultLambdaApproximation, sizeof(double)*nCols);

            // --- Copy Values to Device

            cudaMemcpy(d_I, I, sizeof(double)*nCols*nRows, cudaMemcpyHostToDevice);
            cudaMemcpy(d_eigenValues, lambdaVector, sizeof(double)*nCols, cudaMemcpyHostToDevice);

            // Start the algorithm for finding, given each eigenvalues the associated eigenvector
            
            cout << "Start finding the associate eigenvectors.." << endl;
            fill(&O[0],&O[nCols*nCols],0);
            for (int u=0;u<nEigenValues;u++){
                // Initialize all the matrices that are needed for computation
                cudaMemset(d_Mi_Inverse, 0, sizeof(double)*nCols*nCols);
                cudaMemset(d_Mi_Cholesky, 0, sizeof(double)*nCols*nCols);
                cudaMemset(d_Mi_CholeskyInverse, 0, sizeof(double)*nCols*nCols) ;
                cudaMemset(d_Mi_CholeskyInverse_Transpose, 0, sizeof(double)*nCols*nCols) ;
                
                // Choose a certain value for the eigenvalue that allow us to find a good solution in reasonable time

                double eigenValue = (u==nCols-1) ? 0 : (lambdaVector[u] - abs(((lambdaVector[u] - lambdaVector[u+1])/4)));

                // Initialization of RHS vector 
                for(int i=0;i<nCols;i++)
                    hVk[i] = 1;
                cudaMemcpy(d_Vk, hVk, sizeof(double)*nCols, cudaMemcpyHostToDevice);
                
                

                // Using cublas calculate the norm of the vector V(k)

                normalizeVectorNEW<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_Vk,nCols);

                //Calculate Mi as A-(eigenValue*I);
                cudaMemcpy(d_Mi, d_I, sizeof(double)*nCols*nCols, cudaMemcpyDeviceToDevice);
                matrixMCalculation<<<dim3((int)(nCols/threadsSize)+1),dim3(threadsSize)>>>(d_Mi,eigenValue,nCols);
                // cudaMemcpy(h_Mi, d_Mi, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------1--------\n", h_Mi, nCols);
                
                /*
                    Begin the calculation of the Moore-Penrose Pseudo-Inverse
                    X" = (Mi' * Mi)^-1 * Mi', to do this we used this method:
                        1) Find Mi' * Mi (they are symmetric) don't care for the transposition
                        2) Find the cholesky decomposition of this product
                        3) Calculate the inverse of the cholesky upper matrix
                        4) Multiply the inverse of cholesky with its transpose for finding (Mi' * Mi)^-1
                        5) Multiply the result with M' that is symmetric so the transposition doesn't care
                */

                // Step 1

                matrixSelfMulNaive<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_Mi,d_Mi_SelfMultiplied,nCols,nCols);
                // cudaMemcpy(h_Mi, d_Mi_SelfMultiplied, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------2--------\n", h_Mi, nCols);
                // cin >> a;
                // Step 2
                
                choleskyMatrixCalculation<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_Mi_SelfMultiplied,d_Mi_Cholesky,nCols);
                // cudaMemcpy(h_Mi, d_Mi_Cholesky, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------3--------\n", h_Mi, nCols);
                // cin >> a;
                // Step 3
                
                inverseCholesky<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_Mi_Cholesky,d_Mi_CholeskyInverse,nCols);
                // cudaMemcpy(h_Mi, d_Mi_CholeskyInverse, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------4--------\n", h_Mi, nCols);
                // cin >> a;
                // Step 4
                
                transpose<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_Mi_CholeskyInverse,d_Mi_CholeskyInverse_Transpose,nCols);
                // cudaMemcpy(h_Mi, d_Mi_CholeskyInverse_Transpose, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------5--------\n", h_Mi, nCols);
                // cin >> a;
                // Step 5
                
                matrixSelfMulNaive<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_Mi_CholeskyInverse_Transpose,d_Mi_MoorePenrose,nCols,nCols);
                // cudaMemcpy(h_Mi, d_Mi_MoorePenrose, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                // printMatrix("--------6--------\n", h_Mi, nCols);
                // cin >> a;
                // Step 6
                
                matrix_multiply2<<<dim3((int)(nCols/threadsSize)+1, (int)(nCols/threadsSize)+1),dim3(threadsSize,threadsSize)>>>(d_Mi_MoorePenrose,d_Mi,d_Mi_Transient,nCols);
                // cudaMemcpy(h_Mi, d_Mi_Transient, sizeof(double)*nRows*nCols, cudaMemcpyDeviceToHost);
                
                // printMatrix("--------7--------\n", h_Mi, nCols);
                // cin >> a;
                              
                /* After the calculation of the pseudo-inverse we are able to find the solution of the liner system as following:
                      x(k) = X"*v(k)
                    We apply the power inverse algorithm for finding each eigenvector related to each eigenvalue, the stopping
                    criteria that we choose are:
                    1) The eigenvalue that we find during the computation changes during the iteration no more than a given threshold
                    2) The absolute difference of the reference eigenvalue and the calculated ones is less than a given threshold
                */
                
                do{
                    hApproximatedLambda_pre = hApproximatedLambda;
                    
                    cudaMemset(d_LambdaEstimated,0,sizeof(double));
                    cudaMemset(d_X,0,nCols*sizeof(double));
                    cudaMemset(partialResultLambdaApproximation,0,nCols*sizeof(double));
                    
                    // Calculate the solution vector of the linear system
                    
                    eigenvalueEstimate1<<<dim3(1 + (nCols - 1) / (nCols < 1024 ? nCols : 1024), nCols),dim3(nCols < 1024 ? nCols : 1024, 1)>>>(d_X,d_Mi_Transient,d_Vk,nCols);
                    
                    
                    // Define the RHS vector as the normalized vector of the solution

                    normalizeVectorNEW<<<dim3(1),dim3(nCols < 1024 ? nCols : 1024)>>>(d_X,nCols);
                    
                    cudaMemcpy(d_Vk, d_X, nCols*sizeof(double), cudaMemcpyDeviceToDevice);


                    eigenvalueEstimate1<<<dim3(1 + (nCols - 1) / (nCols < 1024 ? nCols : 1024), nCols),dim3(nCols < 1024 ? nCols : 1024, 1)>>>(partialResultLambdaApproximation,d_I,d_Vk,nCols);
                    
                    eigenvalueEstimate2<<<dim3(1+(nCols-1)/(nCols < 1024 ? nCols : 1024)),dim3(nCols < 1024 ? nCols : 1024)>>>(d_LambdaEstimated,d_Vk,partialResultLambdaApproximation,nCols);
                    cudaMemcpy(&hApproximatedLambda, d_LambdaEstimated, sizeof(double), cudaMemcpyDeviceToHost);
                    // cout << hApproximatedLambda <<","<< lambdaVector[u] <<endl;
                    // cout << abs(hApproximatedLambda - lambdaVector[u]) << endl;
                    // cin >> a;

                }while((abs(hApproximatedLambda - hApproximatedLambda_pre) > threshold) && (abs(hApproximatedLambda - lambdaVector[u]) > threshold));
                cudaMemcpy(&O[u*nCols], d_X, sizeof(double)*nCols, cudaMemcpyDeviceToHost);   
                cout << u+1 <<" of "<< nEigenValues <<endl;
            }
            cout << "Finished" << endl;
            cudaFree(d_Vk);
            cudaFree(d_Mi);
            cudaFree(d_Mi_MoorePenrose);
            cudaFree(d_Mi_SelfMultiplied);
            cudaFree(d_Mi_Inverse);
            cudaFree(d_Mi_Cholesky);
            cudaFree(d_Mi_CholeskyInverse);
            cudaFree(d_Mi_CholeskyInverse_Transpose);
            cudaFree(d_I);
            cudaFree(d_eigenValues);
        }

        void startClock(int step, unsigned int row, double matrix[][4]){
            struct timeval nowStruct;
            gettimeofday(&nowStruct, NULL);
            
            double nowDouble = nowStruct.tv_sec + nowStruct.tv_usec / 1e6;
            unsigned int col = step - 1;
            
            matrix[row][col]=nowDouble;
        }
        
        void stopClock(int step, unsigned int row, double matrix[][4]){
            struct timeval nowStruct;
            gettimeofday(&nowStruct, NULL);
        
            double nowDouble = nowStruct.tv_sec + nowStruct.tv_usec / 1e6;
            unsigned int col = step - 1;
            
            matrix[row][col] = nowDouble - matrix[row][col];
        }

    private:
        void printMatrix(char *name, double * matrix, unsigned int dim){
            cout << name << endl;
            for(int i=0;i<dim;i++){
                for(int j=0;j<dim;j++){
                    cout << matrix[j + i*dim] << " ";
                }
                cout << endl;
            }
        }
};
