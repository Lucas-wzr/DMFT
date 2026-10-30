//================================================================================
//    Phys. Rev. B, 2007,215:155113. https://doi.org/10.1103/PhysRevB.75.155113
//    Phys. Rev. B, 2010,81:195107. https://doi.org/10.1103/PhysRevB.81.195107
//    The energy unit used is eV
//=================================================================================

//=================================================================================
//    Bundled CT-HYB matches ru-ccmt/EDMFTF commit
//    50cdecdc34064a6276b7fae4b7753a2f768fdb46 (2024-03-31),
//    with local entry-point and MPI lifecycle adaptations
//=================================================================================
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
  class Rutgers_CTHYB
  {
    public:
    Rutgers_CTHYB(){};
    ~Rutgers_CTHYB(){};

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

    void write_params(DMFT::input_info& in, const std::string file, const double mu);
    void write_delta_omega(DMFT::input_info& in, const std::string file,
          const int nspin, const int nomega, 
          const int m_tot, const std::vector<double>& freq,
          const std::vector<std::vector<DFT_plus_DMFT::Matrix>>& hyb_omega);

    void write_cix(const int nspin, const int m_tot, const std::string file, 
        const std::vector<DFT_plus_DMFT::Matrix>& Esplit,
        const std::vector<std::vector<std::vector<std::vector<double>>>>& Utensor);

    //========================interface=========================
    std::vector<std::vector<DFT_plus_DMFT::Matrix>>&
          hopping_term(){return hopping_matrix;}

    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
        hybridization_func_tau(){return hyb_tau;}

    private:
    //hopping_term[ineq][is][m_index]
    std::vector<std::vector<DFT_plus_DMFT::Matrix>> hopping_matrix;

    //hyb_tau[ineq][is][i_tau][m_index];
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>> hyb_tau;

  };
}
