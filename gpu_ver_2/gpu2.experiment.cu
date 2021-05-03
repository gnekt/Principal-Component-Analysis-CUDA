/*
    Program that generate all the experiment needed for the cpu program of PCA
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "math.h"
#include "gpu2.interface.cu"
#include <unistd.h>
using namespace std;



/*
    With Love from Giacomo Nunziati & Christian Di Maio
    How to use it:
    ./experiment <string> name of output csv (Eg. kronos.cpu ) 
*/


int main(int argc, char* argv[]){

    // Read the given csv file

    string line;                    /* string to hold each line */
    ofstream outputFile_handler;
    
    string experimentsFileName[2] = {"soil_data.csv","hapt_train_set.csv"};
    char experimentsFileToken[2] = {',',' '};
    int experimentFeaturesPicking[11][2] = {{3109,10},{3109,15},{3109,20},{3109,32},{7767,10},{7767,50},{7767,100},{7767,200},{7767,300},{7767,400},{7767,561}};
    //int experimentFeaturesPicking[11][2] = {{3109,10},{3109,15},{3109,20},{3109,32},{7767,50},{7767,50},{7767,50},{7767,100},{7767,100},{7767,100},{7767,100}};
    int experimentNumberOfIterationGpu[11] = {10,10,10,10,10,10,10,10,10,10,10};
    int experimentDataSetIteration[11] = {0,0,0,0,1,1,1,1,1,1,1};

    if (argc < 2) { /* validate at least 1 argument given */
        cerr << "error: insufficient input.\n"
                "usage: " << argv[0] << " <string> name of output csv (Eg. kronos.cpu ) \n";
        return 1;
    }

    string outputExperimentFileName = argv[1];

    

    double experimentMeasurement[11][16];

    for (int experiment = 0; experiment < 11; experiment ++){
        string fileName = experimentsFileName[experimentDataSetIteration[experiment]];
        char columnSeparatorToken = experimentsFileToken[experimentDataSetIteration[experiment]];
        int NRows = experimentFeaturesPicking[experiment][0];
        int NCols = experimentFeaturesPicking[experiment][1];
        int n_eigenvalues = 0;

        double interTimeIterations[10][3];
        double* inputMatrix = new double[NRows*NCols];
        double* normalizedMatrix = new double[NRows*NCols];
        double* covarianceMatrix = new double[NCols*NCols];
        double* eigenValuesMatrix = new double[NCols*NCols];
        double* eigenValues = new double[NCols] ;
        double* eigenVectors = new double[NCols*NCols];
        
        double elapsedTimeMean; 
        double elapsedTimeStd;
        double elapsedTimeMin;
        double elapsedTimeMax;

        Interface inter;
        ifstream f (fileName);   /* open file */
            if (!f.is_open()) {     /* validate file open for reading */
                perror (("error while opening file " + fileName).c_str());
                return 1;
            }
            int i=0;
            int j=0;
            while (getline (f, line) && i < NRows) {         /* read each line */
                j=0;
                string val;                     /* string to hold value */
                stringstream s (line);          /* stringstream to parse csv */
                while (getline (s, val, columnSeparatorToken) && j < NCols){   /* for each value */
                    inputMatrix[j + i*NCols] =  stod(val);  /* convert to double, add to row */
                    
                    j++;
                }
               
                i++;
            }
        f.close();
        for (int iteration = 0; iteration < experimentNumberOfIterationGpu[experiment]; iteration++){

            // Step 1 : Normalization of input matrix
            inter.startClock(1, iteration, interTimeIterations);
            inter.matrixNormalizationHost(inputMatrix,normalizedMatrix, NRows, NCols);
            inter.stopClock(1, iteration, interTimeIterations);

            // Step 2 : Retrieve covariance matrix
            inter.startClock(2, iteration, interTimeIterations);
            inter.matrixSelfMultiplication(normalizedMatrix, covarianceMatrix, NRows, NCols);
            inter.stopClock(2, iteration, interTimeIterations);

            // Step 3: Find eigenvalues and eigenvectors of the covariance matrix
            inter.startClock(3, iteration, interTimeIterations);
            n_eigenvalues = inter.jbEigenvaluesFinder(covarianceMatrix, eigenVectors, eigenValues, NCols, NCols);
            inter.stopClock(3, iteration, interTimeIterations);
        }
        for (int stepIdx = 0; stepIdx < 3; stepIdx++){
            elapsedTimeMax = 0; 
            elapsedTimeStd = 0;
            elapsedTimeMean = 0;
            for(int iterationTimeIdx=0; iterationTimeIdx < experimentNumberOfIterationGpu[experiment]; iterationTimeIdx++){
                //check Max
                if (interTimeIterations[iterationTimeIdx][stepIdx] > elapsedTimeMax) elapsedTimeMax = interTimeIterations[iterationTimeIdx][stepIdx];
                //mean
                elapsedTimeMean += (interTimeIterations[iterationTimeIdx][stepIdx]) / experimentNumberOfIterationGpu[experiment];
            }
            elapsedTimeMin = interTimeIterations[0][stepIdx];
            for(int iterationTimeIdx=0; iterationTimeIdx < experimentNumberOfIterationGpu[experiment]; iterationTimeIdx++){
                if (interTimeIterations[iterationTimeIdx][stepIdx] < elapsedTimeMin) elapsedTimeMin = interTimeIterations[iterationTimeIdx][stepIdx];
                
                elapsedTimeStd += ((interTimeIterations[iterationTimeIdx][stepIdx] - elapsedTimeMean)*(interTimeIterations[iterationTimeIdx][stepIdx] - elapsedTimeMean)) / experimentNumberOfIterationGpu[experiment];
            }
            elapsedTimeStd = sqrt(elapsedTimeStd);
            experimentMeasurement[experiment][stepIdx*4] = elapsedTimeMean;
            experimentMeasurement[experiment][1 + stepIdx*4] = elapsedTimeStd;
            experimentMeasurement[experiment][2 + stepIdx*4] = elapsedTimeMin;
            experimentMeasurement[experiment][3 + stepIdx*4] = elapsedTimeMax;
        }

        outputFile_handler.open(outputExperimentFileName + "." +to_string(experiment+1) + ".csv");
            for(int i=0; i < NCols;i++){
                for(int j=0; j<n_eigenvalues-1; j++){
                    outputFile_handler << eigenVectors[i*NCols + (NCols-j-1)] << ",";
                    }
                outputFile_handler << eigenVectors[i*NCols + NCols-n_eigenvalues] << endl;
            }
        outputFile_handler.close();
    }

    outputFile_handler.open(outputExperimentFileName + ".measurement.csv");
    for(int experiment=0; experiment < 11; experiment++){
        for(int stepIdx=0; stepIdx<15; stepIdx ++){
            outputFile_handler << experimentMeasurement[experiment][stepIdx] << ",";
        }
        outputFile_handler << experimentMeasurement[experiment][3 + 3*4] << endl;
    }
    outputFile_handler.close();
    return 0;
}
