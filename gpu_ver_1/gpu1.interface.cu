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

struct jacobiParameters {
    int r;
    int s;
    double aRS;
};

/* ==================== Atomic helpers for double ==================== */

__device__ double atomicAddF(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMaxF(double* address, double val) {
    unsigned long long int* address_as_i = (unsigned long long int*)address;
    unsigned long long int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/* ==================== Normalization kernels ==================== */

__global__ void matrixMeanMax(double *A, int NRows, int NCols, double *max, double *mean) {
    int xidx = blockIdx.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (xidx < NCols && yidx < NRows) {
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

        if (threadIdx.y == 0) {
            atomicAddF(&mean[xidx], sum / NRows);
            atomicMaxF(&max[xidx], local_max);
        }
    }
}

__global__ void matrixNormalize(double *A, double *R, int NRows, int NCols, double *max, double *mean) {
    int xidx = blockIdx.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (xidx < NCols && yidx < NRows) {
        R[xidx + yidx * NCols] = (A[xidx + yidx * NCols] - mean[xidx]) / max[xidx];
    }
}

/* ==================== Matrix multiplication kernels ==================== */

__global__ void matrixSelfMulNaive(double *A, double *C, int NRows, int NCols) {
    // Computes C = A^T * A (covariance-like product)
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (xidx < NCols && yidx < NCols) {
        double partialSum = 0;
        for (int i = 0; i < NRows; i++) {
            partialSum += A[yidx + i * NCols] * A[xidx + i * NCols];
        }
        C[xidx + yidx * NCols] = partialSum;
    }
}

__global__ void matrix_multiply2(double *a, double *b, double *ab, unsigned int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= width || col >= width) return;
    float result = 0;

    for (int k = 0; k < width; ++k) {
        result += a[row * width + k] * b[k * width + col];
    }
    ab[row * width + col] = result;
}

/* ==================== Jacobi eigenvalue kernels ==================== */

__global__ void jacobiFindMax(double *A, int NRows, jacobiParameters *var) {
    /*
        Find the maximum absolute value in the upper extra-diagonal triangle of A.
        grid = dim3((nCols-1)/ts+1, (nCols-1)/ts+1), block = dim3(ts, ts), ts <= 32
    */
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ double local_max;

    if (xidx >= NRows || yidx >= xidx) return;

    double val = A[xidx + yidx * NRows];
    val = val < 0 ? -val : val;

    if (threadIdx.y == 0 && (threadIdx.x == ((blockIdx.x == blockIdx.y) ? 1 : 0))) {
        local_max = 0;
    }
    __syncthreads();

    atomicMaxF(&local_max, val);
    __syncthreads();

    if (threadIdx.y == 0 && (threadIdx.x == ((blockIdx.x == blockIdx.y) ? 1 : 0))) {
        atomicMaxF(&(var->aRS), local_max);
    }
}

__global__ void jacobiFindRS(double *A, int NRows, jacobiParameters *var) {
    /*
        Find the (r, s) indices of the max element found by jacobiFindMax.
        grid/block same as jacobiFindMax.
    */
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (xidx >= NRows || yidx >= xidx) return;

    double val = A[xidx + yidx * NRows];
    double max = var->aRS;
    double compareV = (max - (val < 0 ? -val : val)) / max;

    if (compareV < 1e-7) {
        var->r = yidx;
        var->s = xidx;
        var->aRS = val;
    }
}

__global__ void jacobiIteration(double *A, int NRows, jacobiParameters *var) {
    /*
        Perform one Jacobi rotation: A' = G_rs^T * A * G_rs
        grid = dim3(1), block = dim3(nCols)
    */
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (xidx >= NRows) return;

    __shared__ int shared_r;
    __shared__ int shared_s;
    __shared__ double cosFi;
    __shared__ double sinFi;
    __shared__ double Arr;
    __shared__ double Ass;
    __shared__ double Ars;

    if (xidx == 0) {
        shared_r = var->r;
        shared_s = var->s;
        Arr = A[shared_r + shared_r * NRows];
        Ass = A[shared_s + shared_s * NRows];
        Ars = A[shared_s + shared_r * NRows];

        double m = (A[var->r + var->r * NRows] - A[var->s + var->s * NRows]) / (2 * var->aRS);
        double t = -m + ((m >= 0) ? sqrt(1 + m * m) : (-sqrt(1 + m * m)));
        cosFi = 1 / (sqrt(1 + t * t));
        sinFi = t * cosFi;
    }
    __syncthreads();

    double Air = A[shared_r + xidx * NRows];
    double Ais = A[shared_s + xidx * NRows];

    double AisNew = Ais * cosFi - Air * sinFi;
    double AirNew = Air * cosFi + Ais * sinFi;

    A[shared_r + xidx * NRows] = AirNew;
    A[shared_s + xidx * NRows] = AisNew;

    // symmetric update
    A[xidx + shared_r * NRows] = AirNew;
    A[xidx + shared_s * NRows] = AisNew;

    __syncthreads();

    if (xidx == 0) {
        double ArrNew = Arr * cosFi * cosFi + 2 * Ars * cosFi * sinFi + Ass * sinFi * sinFi;
        double AssNew = Ass * cosFi * cosFi - 2 * Ars * cosFi * sinFi + Arr * sinFi * sinFi;
        A[shared_r + shared_r * NRows] = ArrNew;
        A[shared_s + shared_s * NRows] = AssNew;
        A[shared_s + shared_r * NRows] = 0;
        A[shared_r + shared_s * NRows] = 0;
    }
}

/* ==================== Eigenvector kernels ==================== */

__global__ void matrixMCalculation(double *Mi, double eigenValue, int NCols) {
    // Compute Mi = A - eigenValue * I (diagonal shift, in-place)
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (xidx >= NCols) return;
    Mi[xidx + xidx * NCols] = Mi[xidx + xidx * NCols] - eigenValue;
}

__global__ void transpose(double *I, double *O, unsigned int dim) {
    int xidx = threadIdx.x + blockIdx.x * blockDim.x;
    int yidx = threadIdx.y + blockIdx.y * blockDim.y;
    if (xidx >= dim || yidx >= dim) return;
    O[xidx + yidx * dim] = I[yidx + xidx * dim];
}

__global__ void choleskyMatrixCalculation(double *I, double *O, unsigned int dim) {
    // Cholesky decomposition: O = upper triangular factor of I
    int bd = blockDim.x;
    int tx = threadIdx.x;
    double partialSum;
    int height = ((int)((dim - 1) / bd)) * bd + tx;

    for (int row = 0; row <= height; row++) {
        for (int col = tx; col < dim; col += bd) {
            partialSum = I[col + row * dim];
            if (row == col) {
                for (int k = 0; k <= row - 1; k++) {
                    partialSum -= O[col + k * dim] * O[col + k * dim];
                }
                O[col + row * dim] = sqrt(partialSum);
            }
            __syncthreads();
            if (row != col) {
                for (int k = 0; k <= row - 1; k++) partialSum -= O[row + k * dim] * O[col + k * dim];
                O[col + row * dim] = partialSum / O[row + row * dim];
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // zero out lower triangle
    for (int col = tx; col < dim; col += bd) {
        for (int i = col + 1; i < dim; i++) O[i * dim + col] = 0;
    }
}

__global__ void inverseCholesky(double *L, double *R, unsigned int dim) {
    // Invert upper triangular matrix L into R
    int bd = blockDim.x;
    int tx = threadIdx.x;

    for (int i = tx; i < dim; i += bd) {
        R[i + i * dim] = 1 / L[i + i * dim];
    }
    __syncthreads();

    for (int diag = 0; diag < dim; diag++) {
        for (int row = tx; row < dim; row += bd) {
            int col = row + diag + 1;
            if (col >= dim) break;
            double sum = 0;
            for (int j = row + 1; j <= col; j++) sum -= L[j + row * dim] * R[j * dim + col];
            R[row * dim + col] = R[row * dim + row] * sum;
        }
        __syncthreads();
    }
    for (int i = 0; i < tx; i++) R[tx * dim + i] = 0;
}

__global__ void normalizeVector(double *v, unsigned int dim) {
    // In-place L2 normalization
    int tx = threadIdx.x;
    int bdim = blockDim.x;

    __shared__ double norm;
    double partialSum = 0;

    if (tx == 0) norm = 0;
    __syncthreads();

    for (int xidx = tx; xidx < dim; xidx += bdim) {
        double val = v[xidx];
        partialSum += (val * val);
    }
    atomicAddF(&norm, partialSum);
    __syncthreads();

    if (tx == 0) norm = sqrt(norm);
    __syncthreads();

    for (int xidx = tx; xidx < dim; xidx += bdim) {
        v[xidx] = v[xidx] / norm;
    }
}

__global__ void eigenvalueEstimate1(double *partialResult, double *A, double *v, unsigned int dim) {
    // Compute partialResult = A * v (matrix-vector product)
    int tx = threadIdx.x;
    int bx = blockIdx.x, row = blockIdx.y;
    int bdx = blockDim.x;
    int col = tx + bx * bdx;

    __shared__ double partialSum;
    if (col >= dim) return;
    if (tx == 0) partialSum = 0;
    __syncthreads();

    double val = v[col] * A[col + row * dim];
    atomicAddF(&partialSum, val);
    __syncthreads();

    if (tx == 0) atomicAddF(&partialResult[row], partialSum);
}

__global__ void eigenvalueEstimate2(double *result, double *partialResult, double *v, unsigned int dim) {
    // Compute result = v^T * partialResult (Rayleigh quotient step)
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int bdx = blockDim.x;

    __shared__ double partialSum;
    if (tx == 0) partialSum = 0;
    __syncthreads();

    int idx = tx + bx * bdx;
    if (idx >= dim) return;
    double val = partialResult[idx] * v[idx];
    atomicAddF(&partialSum, val);
    __syncthreads();

    if (tx == 0) atomicAddF(result, partialSum);
}

/* ==================== Sorting / reduction kernels ==================== */

__global__ void vectorOrder(double *in, unsigned int dim) {
    // Parallel bubble sort (descending)
    int tx = threadIdx.x;
    int bd = blockDim.x;

    __shared__ unsigned int sorted_flag;
    int n_values = (int)((dim - 1) / (bd)) + 1;
    if (n_values % 2 == 1) n_values++;
    int start = tx * n_values;
    if (start >= dim) return;
    int end = start + n_values;
    if (end - 1 >= dim) end = dim;

    do {
        // intra-thread swap pass
        for (int i = start; i < end - 1; i++) {
            double prev = in[i];
            double next = in[i + 1];
            if (next > prev) {
                in[i] = next;
                in[i + 1] = prev;
            }
        }
        __syncthreads();

        // inter-thread boundary swap
        if (tx != 0) {
            double prev = in[start - 1];
            double next = in[start];
            if (next > prev) {
                in[start - 1] = next;
                in[start] = prev;
            }
        }

        if (tx == 0) sorted_flag = 0;
        __syncthreads();

        // check if sorted
        for (int i = ((start == 0) ? (start + 1) : start); i < end; i++) {
            double prev = in[i - 1];
            double next = in[i];
            if (prev > next) continue;
            atomicAdd(&sorted_flag, 1);
        }
        __syncthreads();

    } while (sorted_flag > 0);
}

__global__ void vectorSum(double *input, double *result, unsigned int width) {
    int tx = threadIdx.x;
    int bd = blockDim.x;

    if (tx >= width) return;

    __shared__ double sum;
    double localSum = 0;
    if (tx == 0) sum = 0;

    for (int i = tx; i < width; i += bd) localSum += input[i];
    __syncthreads();

    atomicAddF(&sum, localSum);
    __syncthreads();

    if (tx == 0) *result = sum;
}

/* ==================== Interface class ==================== */

class Interface {
    public:

        void matrixNormalizationHost(double *I, double *O, int nRows, int nCols, unsigned int threadsSize = 1024) {
            /*
                Normalize input matrix column-wise on GPU:
                    O[col, row] = (I[col, row] - mean[col]) / max[col]
            */
            double *d_I, *d_O, *max, *mean;

            cudaMalloc((void**)&d_I, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_O, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&max, sizeof(double) * nCols);
            cudaMalloc((void**)&mean, sizeof(double) * nCols);

            cudaMemcpy(d_I, I, sizeof(double) * nCols * nRows, cudaMemcpyHostToDevice);
            cudaMemset(max, 0, sizeof(double) * nCols);
            cudaMemset(mean, 0, sizeof(double) * nCols);

            matrixMeanMax<<<dim3(nCols, (int)(nRows / threadsSize + 1)), dim3(1, threadsSize)>>>(d_I, nRows, nCols, max, mean);
            matrixNormalize<<<dim3(nCols, (int)(nRows / threadsSize + 1)), dim3(1, threadsSize)>>>(d_I, d_O, nRows, nCols, max, mean);

            cudaMemcpy(O, d_O, sizeof(double) * nCols * nRows, cudaMemcpyDeviceToHost);

            cudaFree(d_I);
            cudaFree(d_O);
            cudaFree(max);
            cudaFree(mean);
        }

        void matrixSelfMultiplication(double *I, double *O, int nRows, int nCols, unsigned int threadsSize = 32) {
            // Compute O = I^T * I on GPU
            double *d_I, *d_O;

            cudaMalloc((void**)&d_I, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_O, sizeof(double) * nCols * nCols);
            cudaMemcpy(d_I, I, sizeof(double) * nCols * nRows, cudaMemcpyHostToDevice);

            matrixSelfMulNaive<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_I, d_O, nRows, nCols);

            cudaMemcpy(O, d_O, sizeof(double) * nCols * nCols, cudaMemcpyDeviceToHost);
            cudaFree(d_O);
            cudaFree(d_I);
        }

        int jbEigenvaluesFinder(double *I, double *O, int nRows, int nCols, unsigned int threadsSize = 32, double threshold = 1e-3) {
            /*
                Find eigenvalues via Jacobi iteration on GPU.
                Returns the number of principal components capturing 99.9% of variance.
            */
            double percentileSum = 0;
            double* tempMatrix = new double[nCols * nCols];
            double* h_EigenValues = new double[nCols];
            double hARS;
            int a = 0;
            double result;
            double *d_I, *d_EigenValues, *d_result;
            jacobiParameters *jParameters;

            cudaMalloc((void**)&d_I, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&jParameters, sizeof(jacobiParameters));
            cudaMalloc((void**)&d_EigenValues, sizeof(double) * nCols);
            cudaMalloc((void**)&d_result, sizeof(double));

            cudaMemset(d_EigenValues, 0, sizeof(double) * nCols);
            cudaMemset(jParameters, 0, sizeof(jacobiParameters));
            cudaMemset(d_result, 0, sizeof(double));
            cudaMemcpy(d_I, I, sizeof(double) * nCols * nRows, cudaMemcpyHostToDevice);

            jacobiFindMax<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_I, nRows, jParameters);
            jacobiFindRS<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_I, nRows, jParameters);

            cout << "Start Eigenvalues iteration.." << endl;
            do {
                jacobiIteration<<<dim3(1), dim3(nCols)>>>(d_I, nRows, jParameters);

                cudaMemset(jParameters, 0, sizeof(jacobiParameters));
                jacobiFindMax<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_I, nRows, jParameters);
                jacobiFindRS<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_I, nRows, jParameters);

                cudaMemcpy(&hARS, &jParameters->aRS, sizeof(double), cudaMemcpyDeviceToHost);
                a++;
                if (a % 4000 == 0) cout << "Iteration Counter: " << a << ", Max of extradiagonal: " << hARS << endl;
            } while ((hARS > 0 ? hARS : -hARS) > threshold);

            cout << "Stop criterion reached after " << a << " iterations" << endl;
            cout << "Start copying eigenvalues... ";
            cudaMemcpy(tempMatrix, d_I, sizeof(double) * nCols * nCols, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            cout << "DONE" << endl;

            for (int i = 0; i < nCols; i++) {
                h_EigenValues[i] = tempMatrix[i + i * nCols];
            }
            cudaMemcpy(d_EigenValues, h_EigenValues, sizeof(double) * nCols, cudaMemcpyHostToDevice);

            cout << "Start sorting eigenvalues... ";
            vectorOrder<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_EigenValues, nCols);
            cudaDeviceSynchronize();
            cout << "DONE" << endl;

            cout << "Start computing the sum... ";
            vectorSum<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_EigenValues, d_result, nCols);
            cudaDeviceSynchronize();
            cout << "DONE" << endl;

            cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(O, d_EigenValues, sizeof(double) * nCols, cudaMemcpyDeviceToHost);

            for (a = 0; a < nCols; a++) {
                percentileSum += (O[a]) / result;
                if (percentileSum > 0.999) break;
            }
            cout << "The 99% information is stored in the first: " << a << " components." << endl;
            cout << "The others will be zeroed. :)" << endl;

            delete[] tempMatrix;
            delete[] h_EigenValues;
            cudaFree(d_EigenValues);
            cudaFree(d_I);
            cudaFree(jParameters);
            cudaFree(d_result);

            return a;
        }

        void eigenvectorsFinder(double *I, double *O, double *lambdaVector, int nRows, int nCols, unsigned int nEigenValues, unsigned int threadsSize = 32, double threshold = 1e-5) {
            /*
                Find eigenvectors via inverse power iteration with Moore-Penrose pseudo-inverse.
                Steps per eigenvalue:
                    1) M = A - lambda*I
                    2) M+ = (M^T*M)^-1 * M^T  (via Cholesky)
                    3) Power iteration: x(k+1) = M+ * v(k), normalized
                    4) Rayleigh quotient for convergence check
            */
            double *hVk = new double[nCols];
            double hApproximatedLambda = 0;
            double hApproximatedLambda_pre = 0;

            // Device variables
            double *d_eigenValues, *d_I, *d_Vk, *d_Mi;
            double *d_Mi_SelfMultiplied, *d_Mi_Inverse;
            double *d_Mi_Cholesky, *d_Mi_CholeskyInverse, *d_Mi_CholeskyInverse_Transpose;
            double *d_Mi_MoorePenrose, *d_Mi_Transient;
            double *partialResultLambdaApproximation, *d_X, *d_LambdaEstimated;

            cudaMalloc((void**)&d_LambdaEstimated, sizeof(double));
            cudaMalloc((void**)&d_X, nCols * sizeof(double));
            cudaMalloc((void**)&d_Vk, sizeof(double) * nCols);
            cudaMalloc((void**)&d_Mi, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_Mi_MoorePenrose, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_Mi_SelfMultiplied, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_Mi_Inverse, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_Mi_Cholesky, sizeof(double) * nRows * nRows);
            cudaMalloc((void**)&d_Mi_CholeskyInverse, sizeof(double) * nRows * nRows);
            cudaMalloc((void**)&d_Mi_CholeskyInverse_Transpose, sizeof(double) * nRows * nRows);
            cudaMalloc((void**)&d_Mi_Transient, sizeof(double) * nRows * nRows);
            cudaMalloc((void**)&d_I, sizeof(double) * nCols * nRows);
            cudaMalloc((void**)&d_eigenValues, sizeof(double) * nCols);
            cudaMalloc((void**)&partialResultLambdaApproximation, sizeof(double) * nCols);

            cudaMemcpy(d_I, I, sizeof(double) * nCols * nRows, cudaMemcpyHostToDevice);
            cudaMemcpy(d_eigenValues, lambdaVector, sizeof(double) * nCols, cudaMemcpyHostToDevice);

            cout << "Start finding the associate eigenvectors.." << endl;
            fill(&O[0], &O[nCols * nCols], 0);

            for (unsigned int u = 0; u < nEigenValues; u++) {
                cudaMemset(d_Mi_Inverse, 0, sizeof(double) * nCols * nCols);
                cudaMemset(d_Mi_Cholesky, 0, sizeof(double) * nCols * nCols);
                cudaMemset(d_Mi_CholeskyInverse, 0, sizeof(double) * nCols * nCols);
                cudaMemset(d_Mi_CholeskyInverse_Transpose, 0, sizeof(double) * nCols * nCols);

                // Shift eigenvalue for better convergence
                double eigenValue = (u == (unsigned int)(nCols - 1))
                    ? 0
                    : (lambdaVector[u] - abs(((lambdaVector[u] - lambdaVector[u + 1]) / 4)));

                // Initialize RHS vector
                for (int i = 0; i < nCols; i++) hVk[i] = 1;
                cudaMemcpy(d_Vk, hVk, sizeof(double) * nCols, cudaMemcpyHostToDevice);

                normalizeVector<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_Vk, nCols);

                // Mi = A - eigenValue * I
                cudaMemcpy(d_Mi, d_I, sizeof(double) * nCols * nCols, cudaMemcpyDeviceToDevice);
                matrixMCalculation<<<dim3((int)(nCols / threadsSize) + 1), dim3(threadsSize)>>>(d_Mi, eigenValue, nCols);

                /*
                    Moore-Penrose Pseudo-Inverse: M+ = (M^T*M)^-1 * M^T
                    1) M^T * M
                    2) Cholesky decomposition
                    3) Inverse of Cholesky
                    4) (Chol^-1) * (Chol^-1)^T = (M^T*M)^-1
                    5) (M^T*M)^-1 * M = M+
                */
                matrixSelfMulNaive<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_Mi, d_Mi_SelfMultiplied, nCols, nCols);
                choleskyMatrixCalculation<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_Mi_SelfMultiplied, d_Mi_Cholesky, nCols);
                inverseCholesky<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_Mi_Cholesky, d_Mi_CholeskyInverse, nCols);
                transpose<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_Mi_CholeskyInverse, d_Mi_CholeskyInverse_Transpose, nCols);
                matrixSelfMulNaive<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_Mi_CholeskyInverse_Transpose, d_Mi_MoorePenrose, nCols, nCols);
                matrix_multiply2<<<dim3((int)(nCols / threadsSize) + 1, (int)(nCols / threadsSize) + 1), dim3(threadsSize, threadsSize)>>>(d_Mi_MoorePenrose, d_Mi, d_Mi_Transient, nCols);

                // Inverse power iteration
                do {
                    hApproximatedLambda_pre = hApproximatedLambda;

                    cudaMemset(d_LambdaEstimated, 0, sizeof(double));
                    cudaMemset(d_X, 0, nCols * sizeof(double));
                    cudaMemset(partialResultLambdaApproximation, 0, nCols * sizeof(double));

                    // x(k) = M+ * v(k)
                    eigenvalueEstimate1<<<dim3(1 + (nCols - 1) / (nCols < 1024 ? nCols : 1024), nCols), dim3(nCols < 1024 ? nCols : 1024, 1)>>>(d_X, d_Mi_Transient, d_Vk, nCols);

                    normalizeVector<<<dim3(1), dim3(nCols < 1024 ? nCols : 1024)>>>(d_X, nCols);
                    cudaMemcpy(d_Vk, d_X, nCols * sizeof(double), cudaMemcpyDeviceToDevice);

                    // Rayleigh quotient: lambda_est = v^T * A * v
                    eigenvalueEstimate1<<<dim3(1 + (nCols - 1) / (nCols < 1024 ? nCols : 1024), nCols), dim3(nCols < 1024 ? nCols : 1024, 1)>>>(partialResultLambdaApproximation, d_I, d_Vk, nCols);
                    eigenvalueEstimate2<<<dim3(1 + (nCols - 1) / (nCols < 1024 ? nCols : 1024)), dim3(nCols < 1024 ? nCols : 1024)>>>(d_LambdaEstimated, d_Vk, partialResultLambdaApproximation, nCols);
                    cudaMemcpy(&hApproximatedLambda, d_LambdaEstimated, sizeof(double), cudaMemcpyDeviceToHost);

                } while ((abs(hApproximatedLambda - hApproximatedLambda_pre) > threshold) && (abs(hApproximatedLambda - lambdaVector[u]) > threshold));

                cudaMemcpy(&O[u * nCols], d_X, sizeof(double) * nCols, cudaMemcpyDeviceToHost);
                cout << u + 1 << " of " << nEigenValues << endl;
            }
            cout << "Finished" << endl;

            delete[] hVk;
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
            cudaFree(d_Mi_Transient);
            cudaFree(d_X);
            cudaFree(d_LambdaEstimated);
            cudaFree(partialResultLambdaApproximation);
        }

        void startClock(int step, unsigned int row, double matrix[][4]) {
            struct timeval nowStruct;
            gettimeofday(&nowStruct, NULL);
            double nowDouble = nowStruct.tv_sec + nowStruct.tv_usec / 1e6;
            matrix[row][step - 1] = nowDouble;
        }

        void stopClock(int step, unsigned int row, double matrix[][4]) {
            struct timeval nowStruct;
            gettimeofday(&nowStruct, NULL);
            double nowDouble = nowStruct.tv_sec + nowStruct.tv_usec / 1e6;
            matrix[row][step - 1] = nowDouble - matrix[row][step - 1];
        }
};
