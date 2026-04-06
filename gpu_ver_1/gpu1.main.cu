#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "gpu1.interface.cu"
using namespace std;



/*
    With Love from Giacomo Nunziati & Christian Di Maio
    How to use it:
    executable <dataset_csv> <column_separator> <number_of_rows_csv> <number_of_columns_csv>
    eg. executable heart.csv , 314 10
*/

int main(int argc, char* argv[]) {

    string line;
    ofstream outputFile_handler;

    if (argc < 5) {
        cerr << "error: insufficient input.\n"
                "usage: " << argv[0] << " <csv_file> <separator> <nRows> <nCols>\n";
        return 1;
    }

    int NRows = stoi(argv[3]);
    int NCols = stoi(argv[4]);
    double *inputMatrix = new double[NRows * NCols];

    ifstream f(argv[1]);
    if (!f.is_open()) {
        perror(("error while opening file " + string(argv[1])).c_str());
        return 1;
    }

    int i = 0;
    while (getline(f, line) && i < NRows) {
        int j = 0;
        string val;
        stringstream s(line);
        while (getline(s, val, argv[2][0]) && j < NCols) {
            inputMatrix[j + i * NCols] = stod(val);
            j++;
        }
        i++;
    }
    f.close();


    Interface::Interface inter;

    // Step 1 : Normalization of input matrix
    double* normalizedMatrix = new double[NRows * NCols];
    inter.matrixNormalizationHost(inputMatrix, normalizedMatrix, NRows, NCols);

    // Step 2 : Retrieve covariance matrix
    double* covarianceMatrix = new double[NCols * NCols];
    inter.matrixSelfMultiplication(normalizedMatrix, covarianceMatrix, NRows, NCols);

    // Step 3: Find eigenvalues of the training set
    double* eigenValues = new double[NCols];
    int n_eigenvalues = inter.jbEigenvaluesFinder(covarianceMatrix, eigenValues, NCols, NCols);
    cout << "--- EigenValues ---" << endl;
    for (int i = 0; i < n_eigenvalues; i++) {
        cout << "|" << eigenValues[i] << "|" << endl;
    }

    // Step 4: Find associated eigenvectors of the training set
    double* eigenVectors = new double[NCols * NCols];
    inter.eigenvectorsFinder(covarianceMatrix, eigenVectors, eigenValues, NCols, NCols, n_eigenvalues);

    outputFile_handler.open("EigenVectors.csv");
    cout << "Writing eigenvectors.." << endl;
    for (int i = 0; i < NCols; i++) {
        for (int j = 0; j < NCols; j++) {
            outputFile_handler << eigenVectors[i + j * NCols] << ",";
        }
        outputFile_handler << eigenVectors[i + (NCols - 1) * NCols] << endl;
    }
    outputFile_handler.close();

    delete[] inputMatrix;
    delete[] normalizedMatrix;
    delete[] covarianceMatrix;
    delete[] eigenValues;
    delete[] eigenVectors;

    return 0;
}
