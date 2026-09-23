#pragma once

#include <cstddef>
#include <vector>

namespace {
// Assumes 64-byte cache line.
constexpr std::size_t kStrideUnit = 64 / sizeof(double);
} // namespace

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;
  std::vector<double> data_;

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols),
        stride_(((cols + kStrideUnit - 1) / kStrideUnit) * kStrideUnit),
        data_(rows * stride_, 0.0) {}

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }
  const double *row_view(std::size_t i) const {
    return data_.data() + i * stride_;
  }
  double *row_view_mutable(std::size_t i) { return data_.data() + i * stride_; }

  double &operator()(std::size_t i, std::size_t j) {
    return data_[i * stride_ + j];
  }
  double operator()(std::size_t i, std::size_t j) const {
    return data_[i * stride_ + j];
  }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid &old_grid, Grid &new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  for (std::size_t col{}; col < cols; ++col) {
    new_grid(0, col) = old_grid(0, col);
    new_grid(rows - 1, col) = old_grid(rows - 1, col);
  }
#pragma omp parallel for schedule(static)
  for (std::size_t i = 1; i < rows - 1; ++i) {
    new_grid(i, 0) = old_grid(i, 0);
    new_grid(i, cols - 1) = old_grid(i, cols - 1);

    const double *center_row = old_grid.row_view(i);
    const double *above_row = old_grid.row_view(i - 1);
    const double *below_row = old_grid.row_view(i + 1);
    double *__restrict out_row = new_grid.row_view_mutable(i);
#pragma omp simd
    for (std::size_t j = 1; j < cols - 1; ++j) {
      out_row[j] =
          0.5 * center_row[j] + 0.125 * (above_row[j] + below_row[j] +
                                         center_row[j - 1] + center_row[j + 1]);
    }
  }
}
