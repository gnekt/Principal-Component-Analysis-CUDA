/*
    Program that generate all the experiment needed for the cpu program of PCA
*/

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "math.h"
#include <cmath>
#include "cpu.interface.cpp"
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
    int experimentNumberOfIterationCpu[11] = {10,10,10,10,10,10,10,1,1,1,1};
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

        double interTimeIterations[10][4];
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
        ifstream f (fileName.c_str());   /* open file */
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
		    istringstream is(val);
		    double a;
		    is >> a;
                    inputMatrix[j + i*NCols] =  a;  /* convert to double, add to row */
                    j++;
                }
                i++;
            }
        f.close();
        for (int iteration = 0; iteration < experimentNumberOfIterationCpu[experiment]; iteration++){
            fill(&normalizedMatrix[0],&normalizedMatrix[NRows*NCols],0);
            fill(&covarianceMatrix[0],&covarianceMatrix[NCols*NCols],0);
            fill(&eigenValues[0],&eigenValues[NCols],0);
            fill(&eigenVectors[0],&eigenVectors[NCols*NCols],0);

            // Step 1 : Normalization of input matrix
            inter.startClock(1, iteration, interTimeIterations);
            inter.matrixNormalizationHost(inputMatrix,normalizedMatrix, NRows, NCols);
            inter.stopClock(1, iteration, interTimeIterations);

            // Step 2 : Retrieve covariance matrix
            inter.startClock(2, iteration, interTimeIterations);
            inter.matrixSelfMultiplication(normalizedMatrix, covarianceMatrix, NRows, NCols);
            inter.stopClock(2, iteration, interTimeIterations);

	    // Step 3: Find eigenvalues of the training set
            inter.startClock(3, iteration, interTimeIterations);
            n_eigenvalues = inter.jbEigenvaluesFinder(covarianceMatrix, eigenValues, NCols, NCols);
            inter.stopClock(3, iteration, interTimeIterations);
	        for(int p=0; p<10; p++) cout << eigenValues[p] << endl;
            // Step 4: Find associated eigenvectors of the training set
            inter.startClock(4, iteration, interTimeIterations);
            inter.eigenvectorsFinder(covarianceMatrix, eigenVectors, eigenValues, NCols, NCols, n_eigenvalues);
            inter.stopClock(4, iteration, interTimeIterations);

        }
        for (int stepIdx = 0; stepIdx < 4; stepIdx++){
            elapsedTimeMax = 0; 
            elapsedTimeStd = 0;
            elapsedTimeMean = 0;
            for(int iterationTimeIdx=0; iterationTimeIdx < experimentNumberOfIterationCpu[experiment]; iterationTimeIdx++){
                //check Max
                if (interTimeIterations[iterationTimeIdx][stepIdx] > elapsedTimeMax) elapsedTimeMax = interTimeIterations[iterationTimeIdx][stepIdx];
                //mean
                elapsedTimeMean += (interTimeIterations[iterationTimeIdx][stepIdx]) / experimentNumberOfIterationCpu[experiment];
            }
            elapsedTimeMin = interTimeIterations[0][stepIdx];
            for(int iterationTimeIdx=0; iterationTimeIdx < experimentNumberOfIterationCpu[experiment]; iterationTimeIdx++){
                if (interTimeIterations[iterationTimeIdx][stepIdx] < elapsedTimeMin) elapsedTimeMin = interTimeIterations[iterationTimeIdx][stepIdx];
                
                elapsedTimeStd += ((interTimeIterations[iterationTimeIdx][stepIdx] - elapsedTimeMean)*(interTimeIterations[iterationTimeIdx][stepIdx] - elapsedTimeMean)) / experimentNumberOfIterationCpu[experiment];
            }
            elapsedTimeStd = sqrt(elapsedTimeStd);
            experimentMeasurement[experiment][stepIdx*4] = elapsedTimeMean;
            experimentMeasurement[experiment][1 + stepIdx*4] = elapsedTimeStd;
            experimentMeasurement[experiment][2 + stepIdx*4] = elapsedTimeMin;
            experimentMeasurement[experiment][3 + stepIdx*4] = elapsedTimeMax;
        }

	string temp;
	stringstream ss1;
	ss1 << experiment+1;
	ss1 >> temp;
	temp = outputExperimentFileName + "." + temp + ".csv";
        outputFile_handler.open(temp.c_str());
            for(int i=0; i < NCols;i++){
                for(int j=0; j<n_eigenvalues-1; j++){
                    outputFile_handler << eigenVectors[i + j * NCols] << ",";
                    }
                outputFile_handler << eigenVectors[i + (n_eigenvalues-1) * NCols] << endl;
            }
        outputFile_handler.close();
    }
    string temp = outputExperimentFileName + ".mesurement.csv";
    outputFile_handler.open(temp.c_str());
    for(int experiment=0; experiment < 11; experiment++){
        for(int stepIdx=0; stepIdx<15; stepIdx ++){
            outputFile_handler << experimentMeasurement[experiment][stepIdx] << ",";
            }
        outputFile_handler << experimentMeasurement[experiment][3 + 3*4] << endl;
    }
    outputFile_handler.close();
    return 0;
}
