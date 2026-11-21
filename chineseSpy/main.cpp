//
// Created by clementzhao on 20/11/2026.
//

#include "main.hpp"
#include <iostream>

Matrix::Matrix(const int rows, const int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f) {

}
int Matrix::getRows() const {
    return rows;
}

int Matrix::getCols() const {
    return cols;
}


int main() {
    Matrix matrix(3, 4);
    
}