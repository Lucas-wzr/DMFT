#pragma once

#include "coulomb_tensor.h"
#include "../matrix_types.h"
#include <iosfwd>
#include <vector>

namespace DMFT
{
  namespace rutgers_atomic
  {
    struct block
    {
      std::vector<int> states;
      Eigen::VectorXd energy;
      DFT_plus_DMFT::RealMatrix eigenvectors; // row-major: occupation state x eigenstate
      std::vector<int> target;          // creation operator targets, zero-based; -1 = absent
      std::vector<DFT_plus_DMFT::RealMatrix> creation; // source eigenstate x target eigenstate
      int N;
      double Sz;
    };

    interaction_tensor read_tensor(std::istream& input, int M, double energy_unit);
    std::vector<block> solve(const std::vector<double>& levels, const interaction_tensor& U);
    void write_cix(std::ostream& output, const std::vector<double>& levels,
                   const interaction_tensor& U, const std::vector<block>& blocks, double energy_unit);
  }
}
