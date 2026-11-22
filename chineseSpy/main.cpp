//
// Created by clementzhao on 20/11/2026.
//


#include <iostream>
#include <ostream>

#include "Matrix.hpp"
#include <stdio.h>

void printMatrixState(Matrix*& dynamicObject, const std::string& message, const std::string &varName);

int main() {
    using namespace std;

    int row, col;
    cin >> row >> col;

    const Matrix matrix(3, 4);
    Matrix *A = nullptr;

    {
        A = new Matrix();
        Matrix B(3, 5);
        auto C = new Matrix(5, 12);
    }

    cout << "outside of the scope, B has been destroyed." <<endl;

    std::cout << A->getElementCount() << '\n';

    // std::cout << "Before 'delete A': \n&A" << &A  << " A:" << A << "\n====";
    printMatrixState(A, "Before 'delete A'", "A");
    delete A;
    printMatrixState(A, "After 'delete A'", "A");
    A = nullptr;
    printMatrixState(A, "After 'set to nullptr'", "A");


    cout<< matrix.getElementCount() << endl;

}

void printMatrixState(Matrix*& dynamicObject, const std::string& message, const std::string &varName) {
    std::cout <<  message << "\n" << varName << ": " << &dynamicObject  <<  "\n" << varName << ": " << dynamicObject << "\n====";
}
