//=================================================================
//    Only density-density interaction
//=================================================================
#pragma once
#include "../matrix_types.h"

#include "../para/KS_bands.h"
#include "../para/correlated_atoms.h"
#include "../para/input.h"
#include "coulomb_tensor.h"

#include <vector>
#include <complex>

namespace DMFT
{
  class IQIST_NARCISSUS
  {
    public:
    IQIST_NARCISSUS(){};
    ~IQIST_NARCISSUS(){};

    void impurities_solving(
          const int char_step,
          const int DMFT_step,
          DFT_output::atoms_info& atom );
                        
    void read_self_energy(
          const int char_step,
          const int DMFT_step, 
          DFT_output::KS_bands& band,
          DMFT::input_info& in, 
          DFT_output::atoms_info& atom,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& Sw);

    void output(const int char_step, const int DMFT_step, 
          const double mu, DMFT::input_info& in, 
          DFT_output::atoms_info& atom, DFT_output::KS_bands& band, const std::vector<double>& freq,
          std::vector<std::vector<DFT_plus_DMFT::Matrix>>& Eimp,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& hyb_omega,
          DMFT::coulomb_tensor& Umat );

    void write_solver_ctqmc_in(
         const std::string file, const int nband,
         DMFT::input_info& in);

    void write_solver_eimp_in(
         const std::string file, 
         std::vector<DFT_plus_DMFT::Matrix>& muvec,
         const int nband,const int symm,
         const int corr_L, const int nspin);

    void write_solver_umat_in(
         const std::string file, 
         std::vector<std::vector<std::vector<
         std::vector<double>>>>& Umat );

    //Interfaces
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
        hybridization_func_tau(){return hyb_tau;}

    private:
    //hyb_tau[ineq][is][i_tau][m_index];
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>> hyb_tau;

  };
}
