#pragma once

#include <iosfwd>
#include <vector>

namespace DMFT
{
  namespace rutgers_atomic
  {
    // H_int = 1/2 sum U[i][j][k][l] c_i^+ c_j^+ c_k c_l, in Hartree.
    // Spin orbitals: all up orbitals followed by all down orbitals.
    typedef std::vector<std::vector<std::vector<std::vector<double>>>> tensor;

    struct block
    {
      std::vector<int> states;
      std::vector<double> energy;
      std::vector<double> eigenvectors; // row-major: occupation state x eigenstate
      std::vector<int> target;          // creation operator targets, zero-based; -1 = absent
      std::vector<std::vector<double>> creation; // source eigenstate x target eigenstate
      int N;
      double Sz;
    };

    tensor read_tensor(std::istream& input, int M, double energy_unit);
    std::vector<block> solve(const std::vector<double>& levels, const tensor& U);
    void write_cix(std::ostream& output, const std::vector<double>& levels,
                   const tensor& U, const std::vector<block>& blocks, double energy_unit);
  }
}
