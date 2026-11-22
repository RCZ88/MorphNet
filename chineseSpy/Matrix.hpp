
#pragma once

#include <vector>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<float> data;

public:
    Matrix(int rows, int cols);
    Matrix();
    ~Matrix();

    int getRows() const;

    int getCols() const;

    int getElementCount() const;

    void printMatrix() const;
};