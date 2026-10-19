#pragma once

#include "../para/correlated_atoms.h"
#include "../para/KS_bands.h"

#include <vector>

namespace DMFT
{
    // H_int = 1/2 sum U[i][j][k][l] c_i^+ c_j^+ c_k c_l, in Hartree.
    // Spin orbitals: all up orbitals followed by all down orbitals.
    using interaction_tensor = std::vector<std::vector<std::vector<std::vector<double>>>>;

  class coulomb_tensor
  {
    public:
    coulomb_tensor(){};
    ~coulomb_tensor(){};

    void update_coulomb_tensor(const int type, DFT_output::atoms_info& atom);

    void out_coulomb_tensor(
              const int type,
              const int char_step,
              const int DMFT_step,
              const int impurity_solver, 
              DFT_output::atoms_info& atom,
              DFT_output::KS_bands& band );

    std::vector<interaction_tensor>& Coulomb_matrix(){return U_matrix;}

    private:

    std::vector<interaction_tensor> U_matrix; // U[ineq][i][j][k][l], in Hartree

  };
}
