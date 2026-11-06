#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace saddle {

class ShapeError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

template <class T>
class MatrixView {
 public:
  MatrixView(T* data, std::size_t rows, std::size_t cols) noexcept
      : data_(data), rows_(rows), cols_(cols) {}

  template <class U>
    requires std::is_same_v<T, const U>
  MatrixView(MatrixView<U> other) noexcept  // NOLINT(google-explicit-constructor)
      : data_(other.data()), rows_(other.rows()), cols_(other.cols()) {}

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] std::size_t size() const noexcept { return rows_ * cols_; }
  [[nodiscard]] T* data() const noexcept { return data_; }

  T& operator()(std::size_t r, std::size_t c) const noexcept { return data_[r * cols_ + c]; }

  [[nodiscard]] T& at(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_) {
      throw ShapeError("index (" + std::to_string(r) + ", " + std::to_string(c) +
                       ") out of bounds for " + std::to_string(rows_) + "x" +
                       std::to_string(cols_) + " view");
    }
    return (*this)(r, c);
  }

 private:
  T* data_;
  std::size_t rows_;
  std::size_t cols_;
};

template <class T>
void require_shape(const MatrixView<T>& view, std::size_t rows, std::size_t cols,
                   std::string_view name) {
  if (view.rows() != rows || view.cols() != cols) {
    throw ShapeError(std::string(name) + ": expected " + std::to_string(rows) + "x" +
                     std::to_string(cols) + ", got " + std::to_string(view.rows()) + "x" +
                     std::to_string(view.cols()));
  }
}

template <class T>
[[nodiscard]] MatrixView<T> make_matrix_view(std::vector<T>& storage, std::size_t rows,
                                             std::size_t cols) {
  if (storage.size() < rows * cols) {
    throw ShapeError("storage of " + std::to_string(storage.size()) + " elements is too small for " +
                     std::to_string(rows) + "x" + std::to_string(cols));
  }
  return MatrixView<T>(storage.data(), rows, cols);
}

}
