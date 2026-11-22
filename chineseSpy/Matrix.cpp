//
// Created by clementzhao on 22/11/2026.
//

#include "Matrix.hpp"
#include <iostream>

using namespace std;

Matrix::Matrix(const int rows, const int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f) {
    cout << "Created Matrix(" << rows << "x" << cols << ")" << endl;
}

Matrix::Matrix()
    :rows(0), cols(0) {
    cout << "Created Empty Matrix" << endl;
}

Matrix::~Matrix() {
    cout << "Deleted Matrix(" << rows << "x" << cols << ")" << endl;
}


int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}

int Matrix::getElementCount() const {
    return rows * cols;
}

void Matrix::printMatrix() const {
    using namespace std;
    cout << "Rows: " << rows << "\nCols: " << cols << "\nTotal Elements: " << getElementCount() << endl;
}
