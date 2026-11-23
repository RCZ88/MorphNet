
#pragma once

#include <vector>

class Matrix {
private:
    int rows;
    int cols;
    std::vector<float> data;
    int rowStrides;
    int colStrides;
public:
    Matrix(int rows, int cols);
    Matrix();
    ~Matrix();
    Matrix(const Matrix& other); // Copy constructor
    Matrix(Matrix&& other) noexcept;      // Move constructor

    int getRows() const;

    int getCols() const;

    int getElementCount() const;

    void printMatrix() const;

    void transform();

    void multiply(const Matrix &other);
};