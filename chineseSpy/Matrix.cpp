//
// Created by clementzhao on 22/11/2026.
//

#include "Matrix.hpp"
#include <iostream>
#include <expected>

using namespace std;

Matrix::Matrix(const int rows, const int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f), rowStrides(cols), colStrides(1) {
    cout << "Created Matrix(" << rows << "x" << cols << ")" << endl;
}

Matrix::Matrix()
    : rows(0), cols(0), rowStrides(0), colStrides(1) {
    cout << "Created Empty Matrix" << endl;
}

Matrix::~Matrix() {
    cout << "Deleted Matrix(" << rows << "x" << cols << ")" << endl;
}

Matrix::Matrix(const Matrix &other) : rows(other.rows), cols(other.cols), rowStrides(other.rowStrides), colStrides(other.colStrides) {
}


Matrix::Matrix(Matrix&& other) noexcept:
    rows(other.rows),
    cols(other.cols),
    data(std::move(other.data)),
    rowStrides(other.rowStrides),
    colStrides(other.colStrides) {

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

std::string Matrix::shapeString() const {
    return  "(" + std::to_string(rows) +  "x" + std::to_string(cols) + ")";
}

Matrix Matrix::transform() const {
    Matrix transform(cols, rows);
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++) {
            transform.data[i * rowStrides + j * colStrides] = data[i * cols + j];
        }
    }
    return transform;
}

expected<Matrix, std::string> Matrix::multiply(const Matrix &other) const {
    if (other.rows != cols) {
        std::string error_msg("Invalid Shape! multiplying" + shapeString()  + " with " + other.shapeString());
        // throw std::invalid_argument("Matrix multiply only supports the same dimension");
        return std::unexpected(error_msg);
    }
    Matrix multiplied(rows, other.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            float sum = 0;
            for (int k = 0; k < cols; k++) {
                // Accumulate A[i][k] * B[k][j]
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            multiplied.data[i * other.cols + j] = sum;
        }
    }

    return multiplied;

}

static expected<Matrix, std::string> operator*(const Matrix &self, const Matrix &other) {
    auto result = self.multiply(other);
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(result.error());

}

expected<Matrix, std::string> Matrix::add(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        return std::unexpected("Error: Shape Missmatch! " + shapeString() + " != " + other.shapeString());
    }
    Matrix added(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            added.data[i * cols + j] += other.data[i * other.cols + j];
        }
    }
    return added;
}
static expected<Matrix, std::string> operator+(const Matrix &self, const Matrix &other) {
    auto result = self.add(other);
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(result.error());
}

expected<Matrix, std::string> Matrix::subtract(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        return std::unexpected("Error: Shape Missmatch! " + shapeString() + " != " + other.shapeString());
    }
    Matrix added(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            added.data[i * cols + j] -= other.data[i * other.cols + j];
        }
    }
    return added;
}

static expected<Matrix, std::string> operator-(const Matrix &self, const Matrix &other) {
    auto result = self.subtract(other);
    if (result.has_value()) {
        return result.value();
    }
    return std::unexpected(result.error());
}
