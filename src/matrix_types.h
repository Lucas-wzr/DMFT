#pragma once

#include <complex>
#include <Eigen/Core>

namespace DFT_plus_DMFT
{
    // Match flattened vector matrices indexed by row * ncol + col.
    using Matrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RealMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}
