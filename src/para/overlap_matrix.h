#pragma once

#include "overlap_matrix_aims.h"
#include "overlap_matrix_abacus.h"
#include "correlated_atoms.h"
#include "../matrix_types.h"

#include <complex>
#include <string>

namespace DFT_output
{
  class overlap_matrix
  {
    public:
    overlap_matrix(){};
    overlap_matrix(const int DFT_solver, const std::string dir);

    ~overlap_matrix(){};

    aims::overlap_matrix ovlp_aims;
    abacus::overlap_matrix ovlp_abacus;

    void evaluate_ovlp_k(const int ik, atoms_info& at_info);

    // Test
    void out();

    // ========================
    //     interface
    //=========================
    DFT_plus_DMFT::Matrix& overlap();
    
    DFT_plus_DMFT::Matrix& overlap_localorb();

    private:
    int flag_DFT_solver;          //1:aims, 2:ABACUS

  };
}
