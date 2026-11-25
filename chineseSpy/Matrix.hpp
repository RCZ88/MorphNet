
#pragma once

#include <vector>
#include <expected>
#include <string>

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

    std::string shapeString() const;

    Matrix transform() const;


    std::expected<Matrix, std::string> multiply(const Matrix &other) const;

    std::expected<Matrix, std::string> add(const Matrix &other) const;

    std::expected<Matrix, std::string> subtract(const Matrix &other) const;
};
