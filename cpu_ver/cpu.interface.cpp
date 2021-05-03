#include <vector>
#include "math.h"
#include <sys/time.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
using namespace std;

class Interface{
    public:
        
        void matrixNormalizationHost(double * I, double * O, int nRows, int nCols){
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
            double *maxs = new double[nCols];
            double *means = new double[nCols];
            
            findMaxMeanForColumn(I,nRows,nCols,maxs,means);


            for(int row=0;row<nRows;row++){
                for(int col=0;col<nCols;col++){
                    O[col + row*nCols] = (I[col + row*nCols] - means[col]) / maxs[col] ;
                }
            }
        }

        

        void matrixSelfMultiplication(double * I, double * O, int nRows, int nCols){
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
           double *tempMatrix = new double[nRows*nCols]; 
           transposeMatrix(I,tempMatrix,nRows,nCols);
           matrixMultiply(tempMatrix,nCols,nRows,I,nRows,nCols,O);
        }
    
        int jbEigenvaluesFinder(double * I, double * O, int nRows, int nCols, double threshold=1e-5){
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
            double max = 0;
            int r = 0, s = 0, iter=0;
            double *operationMatrix = new double[nRows*nRows];
            for(int i=0;i<nRows;i++){
                for(int j=0;j<nRows;j++){
                    operationMatrix[j+i*nRows] = I[j+i*nRows];
                }
            }    
            findMax(operationMatrix,nRows,nRows,&max,&r,&s);
            cout << "Start finding eigenvalues.." << endl;

            do{
                iter++;
                if (iter % 4000 == 0) cout << "Iteration Counter: "<<iter<<", Max of extradiagonal: "<<max<<endl; 
            }while(!jacobiIteration(operationMatrix,nRows,&max,&r,&s,threshold));
            cout << "Stop criterion reached after "<< iter+1 <<" iterations" << endl;

            cout << "Start copying eigenvalues... ";
            double totalSum = 0;
            double *eigenValues = new double [nRows];
            for(int i=0; i<nRows; i++){
                eigenValues[i] = operationMatrix[i + i*nRows];
                totalSum += eigenValues[i];
            }
            cout << "DONE" << endl;

            cout << "Start sorting eivenvalues... ";
            orderVector(eigenValues,nRows,false);
            cout << "DONE" << endl;

            cout << "Start computing the sum... ";
            fill(&O[0],&O[nRows],0);
            int a = 0;
            double percentuleSum = 0;
            for (a=0;a<nRows;++a){
                percentuleSum += (eigenValues[a]) / totalSum;
                O[a] = eigenValues[a];
                if (percentuleSum > 0.999) break;
                
            }
            cout << "DONE" << endl;
            cout << "The 99% information is stored in the first: "<< a <<" components." << endl;
            cout << "The others will be zeroed. :)" << endl;

            return a;
        }
            

        void eigenvectorsFinder(double * I, double * O, double * lambdaVector, int nRows, int nCols,unsigned int nEigenValues, double threshold=1e-5){
            double *Vk = new double[nRows];
            double *Xk = new double[nRows];
            double *matrixMi = new double[nRows*nRows];
            double *matrixMi_selfMultiplied = new double[nRows*nRows];
            double *matrixMi_cholesky = new double[nRows*nRows];
            double *matrixMi_choleskyInverse = new double[nRows*nRows];
            double *matrixMi_CholeskyInverse_Transpose = new double[nRows*nRows];
            double *matrixMi_Transient = new double[nRows*nRows];
            double *matrixMi_MoorePenrose = new double[nRows*nRows];
            double *partialResultLambdaApproximation = new double[nRows];
            
            double lambdaApproximated = 0,previousLambdaApproximated;
            double eigenValue;
            for (int eigen_idx=0; eigen_idx<nEigenValues; eigen_idx++ ){
                eigenValue = (eigen_idx==nCols-1) ? 0 : (lambdaVector[eigen_idx] - fabs(((lambdaVector[eigen_idx] - lambdaVector[eigen_idx+1])/4)));
                lambdaApproximated = 0;
                
                
                
                fill(&matrixMi_selfMultiplied[0],&matrixMi_selfMultiplied[nRows*nRows],0);
                fill(&matrixMi_Transient[0],&matrixMi_Transient[nRows*nRows],0);
                fill(&matrixMi_MoorePenrose[0],&matrixMi_MoorePenrose[nRows*nRows],0);

                // Initialization of RHS vector
                for (int i=0;i<nRows;i++) Vk[i] = 1;
                normalizeVector(Vk,nRows);

                for (int i=0; i<nRows; i++){
                    for (int j=0; j<nRows; j++){
                        matrixMi[j + i*nRows] = (i==j) ? (I[j + i*nRows] - eigenValue) : I[j + i*nRows]; 
                    }
                }
                

                matrixMultiply(matrixMi,nRows,nRows,matrixMi,nRows,nRows,matrixMi_selfMultiplied);

                choleskyMatrixCalculation(matrixMi_selfMultiplied,matrixMi_cholesky,nRows);

                inverseCholeskyMatrixCalculation(matrixMi_cholesky,matrixMi_choleskyInverse,nRows);

                transposeMatrix(matrixMi_choleskyInverse,matrixMi_CholeskyInverse_Transpose,nRows,nRows);
                
                matrixMultiply(matrixMi_choleskyInverse,nRows,nRows,matrixMi_CholeskyInverse_Transpose,nRows,nRows,matrixMi_Transient);

                matrixMultiply(matrixMi_Transient,nRows,nRows,matrixMi,nRows,nRows,matrixMi_MoorePenrose);

                int a;
                do {
                    fill(&Xk[0],&Xk[nRows],0);
                    fill(&partialResultLambdaApproximation[0],&partialResultLambdaApproximation[nRows],0);
                    previousLambdaApproximated = lambdaApproximated;
                    lambdaApproximated = 0;
                    //calculate vk
                    for (int vk_idx = 0; vk_idx < nRows; vk_idx++){
                        for (int col = 0; col<nRows; col++){
                            Xk[vk_idx] += matrixMi_MoorePenrose[col + vk_idx*nRows] * Vk[col];
                        }
                    }
                    normalizeVector(Xk,nRows);
                    
                    for (int vk_idx = 0; vk_idx < nRows; vk_idx++) Vk[vk_idx] = Xk[vk_idx];
                    
                    for (int vk_idx = 0; vk_idx < nRows; vk_idx++){
                        for (int col = 0; col<nRows; col++){
                            partialResultLambdaApproximation[vk_idx] += I[col + vk_idx*nRows] * Vk[col];
                        }   
                        lambdaApproximated += Vk[vk_idx] * partialResultLambdaApproximation[vk_idx];
                    }  
                }while((fabs(lambdaApproximated - previousLambdaApproximated) > threshold) && (fabs(lambdaApproximated - lambdaVector[eigen_idx]) > threshold));
                cout << eigen_idx+1 << " out of " << nEigenValues << endl;
            memcpy (&O[eigen_idx*nCols], Xk, sizeof(double)*nCols);
            }

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
        
        void matrixMultiply (double *a,int a_rows,int a_cols, double *b,int b_rows,int b_cols, double *out){
            for(int a_row=0; a_row<a_rows; a_row++){
                for(int b_col=0; b_col<b_cols; b_col++){
                    for(int a_col=0; a_col < a_cols; a_col++){
                        out[b_col + a_row*b_cols ] += a[a_col + a_row*a_cols] * b[b_col + a_col*b_cols]; 
                    }  
                }
            }
        }

        void findMax(double *in, int in_rows, int in_cols, double *max, int *rowMax, int *colMax, bool upperTriangle = true){
            *max=0;
            *rowMax=-1;
            *colMax =-1;
            double temp_max = 0;
            for(int row=0;row<in_rows;row++){
                for(int col=0;col<in_cols;col++){
                    if (fabs(in[col + row*in_cols]) > fabs(*max) && row<col){
                        *max = in[col + row*in_cols];
                        *rowMax = row;
                        *colMax = col;
                    }
                }
            }
        }

        void transposeMatrix (double *I, double *O, int nRows, int nCols){
            for (int row=0; row<nRows; row++){
                for (int col=0; col<nCols; col++){
                    O[row + col*nRows] = I[col + row*nCols];                 
                }
            }
        }

        void findMaxMeanForColumn(double *I, int nRows, int nCols, double *maxs, double *means){
            fill(&maxs[0],&maxs[nCols],0);
	    fill(&means[0], &means[nCols], 0);
            for(int row=0;row<nRows;row++){
                for(int col=0;col<nCols;col++){
                    if ((I[col + row*nCols]<0 ? -I[col + row*nCols] : I[col + row*nCols]) > maxs[col]){
                        maxs[col] = (I[col + row*nCols]<0 ? -I[col + row*nCols] : I[col + row*nCols]);
                    }
                    means[col] += I[col + row*nCols] / nRows;
		    if(isnan(means[col])){
			cout << "Row: " << row << "; Col: " << col << "; Val: " << I[col + row * nCols] << endl;
			cout << "nRows: " << nRows << "; Val/nRows: " << I[col + row * nCols] / nRows;
			cin.ignore();
		    }
                }
            }
        }

        bool jacobiIteration(double *I, int nRows, double *_max, int *_r, int *_s, double threshold){
            int r = *_r,s = *_s;
            double max = *_max;
            

            double Arr = I[r+r*nRows], Ass = I[s+s*nRows], Ars = I[s+r*nRows];

            double m = (I[r + r * nRows] - I[s + s * nRows]) / (2* max); 
            double t = -m + ((m>=0) ? sqrt(1 + m*m) : (- sqrt(1 + m*m)));    
            double cosFi = 1/(sqrt(1 + t*t)); 
            double sinFi = t * cosFi;

            double Air,Ais,AisNew,AirNew = 0;
            for (int row=0; row<nRows; row++){
                // load the values that will be updated
                Air = I[r + row * nRows];
                Ais = I[s + row * nRows];
                
                // compute the new values
                AisNew = Ais * cosFi - Air * sinFi;
                AirNew = Air * cosFi + Ais * sinFi;

                // update
                I[r + row * nRows] = AirNew;
                I[s + row * nRows] = AisNew;

                // since the resulting matrix will be symmetric, also update the lower triangle
                I[row + r * nRows] = AirNew;
                I[row + s * nRows] = AisNew;
            }
        
            // use their previous value to compute the new values
            double ArrNew = Arr*cosFi*cosFi + 2*Ars*cosFi*sinFi + Ass*sinFi*sinFi;
            double AssNew = Ass*cosFi*cosFi - 2*Ars*cosFi*sinFi + Arr*sinFi*sinFi;
            
            // update the values
            I[r + r * nRows] = ArrNew;
            I[s + s * nRows] = AssNew;

            // if the computation is correct, the extra-diagonal elements are forced to 0, so it can be avoided to compute them
            I[s + r * nRows] = 0;
            I[r + s * nRows] = 0;
            

            
            findMax(I,nRows,nRows,_max,_r,_s);

            return fabs(*_max) < threshold; 
        }

        void orderVector(double *a, unsigned int dim, bool descentOrder){
            double temp=0;
            
            for(int i = 0; i<dim; i++) {
                for(int j = i+1; j<dim; j++)
                {
                    if((a[j] < a[i] && descentOrder) || (a[j] > a[i] && !descentOrder)) {
                        temp = a[i];
                        a[i] = a[j];
                        a[j] = temp;
                    }
                }
            }
        }

        void normalizeVector(double *in, unsigned int dim){
            double totalSum = 0;
            for (int elem = 0; elem < dim; elem++) totalSum += in[elem] * in[elem];
            for (int elem = 0; elem < dim; elem++) in[elem] =  in[elem]/fabs(sqrt(totalSum));
        }

        void choleskyMatrixCalculation(double *in, double *out, unsigned int dim){
            double partialSum;
            for(int col = 0; col < dim ; col++ ){
                for (int row = 0; row <= col ; row++){
                    partialSum = in[col + row*dim];
                    // first of all, the element on the diagonal has to be computed
                    if (row==col){
                        for(int k=0;k<=row-1;k++){
                            partialSum -= out[col + k*dim] * out[col + k*dim];
                        }
                        out[col + row*dim] = sqrt(partialSum);
                    }

                    if(row!=col){
                        for(int k=0;k<=row-1;k++) partialSum -= out[row + k*dim] * out[ col + k*dim];
                        out[col + row*dim] = partialSum / out[row + row*dim];
                    }
                }
            }
            // set to 0 all the terms in the lower, extra-diagonal triangle
            for (int col=0; col<dim; col++){
                for (int row = dim; row > col; row--) out[col + row*dim] = 0;
            }
        }

        void inverseCholeskyMatrixCalculation(double *in, double *out, unsigned int dim){
            for(int col=0; col<dim; col++){
                out[col + col * dim] = 1 / in[col + col * dim];        
                
            }
            for(int col=0; col<dim; col++){
                //loop over the elements associated with the current thread (if dim > 1024)
                for(int diag=0; diag<dim; diag++){
                    for(int row=col; row<dim; row++){
                        int _col=row+diag+1;
                        if(_col>=dim) break;
                        double sum=0;
                        for(int j=row+1; j<=_col; j++) sum-=in[j+row*dim]*out[j*dim+_col];
                        out[row*dim+_col]=out[row*dim+row]*sum;
                    }
                }
                for(int i = 0; i<col; i++) out[col * dim + i] = 0;
            }
        }

        
};
