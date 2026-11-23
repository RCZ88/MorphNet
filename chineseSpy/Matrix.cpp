//
// Created by clementzhao on 22/11/2026.
//

#include "Matrix.hpp"
#include <iostream>

using namespace std;

Matrix::Matrix(const int rows, const int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f), rowStrides(cols), colStrides(1) {
    cout << "Created Matrix(" << rows << "x" << cols << ")" << endl;
}

Matrix::Matrix()
    :rows(0), cols(0) {
    cout << "Created Empty Matrix" << endl;
}

Matrix::~Matrix() {
    cout << "Deleted Matrix(" << rows << "x" << cols << ")" << endl;
}

Matrix::Matrix(const Matrix &other):rows(other.rows), cols(other.cols) {

}

Matrix::Matrix(Matrix&& other) noexcept:
    rows(other.rows),
    cols(other.cols),
    data(std::move(other.data)) {

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

void Matrix::transform()  {
    std::vector<float> transform = std::move(data);
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++) {
            transform[i * rowStrides + j * colStrides] = data[i * cols + j];
        }
    }
}

void Matrix::multiply(const Matrix &other) {
    auto *multiplied = new Matrix(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0;
            for (int k = 0; k < cols; k++) {
                // Accumulate A[i][k] * B[k][j]
                sum += data[i * k + k] * other.data[k * j + j];
            }
            multiplied->data[i * j + j] = sum;
        }
    }

}
