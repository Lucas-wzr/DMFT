//=================================================================
//    Computer Physics Communications 215 (2017) 128–136,
//    https://doi.org/10.1016/j.cpc.2017.01.003
//    The energy unit used is eV
//=================================================================
#pragma once
#include "../matrix_types.h"

#include "../para/KS_bands.h"
#include "../para/correlated_atoms.h"
#include "../para/input.h"

#include <vector>
#include <complex>

namespace DMFT
{
  class ALPS_CTHYB_SEGMENT
  {
    public:
    ALPS_CTHYB_SEGMENT(){};
    ~ALPS_CTHYB_SEGMENT(){};
                        
    void read_last_step(const int istep, 
          DFT_output::KS_bands& band,
          DMFT::input_info& in, 
          DFT_output::atoms_info& atom,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& hyb_save,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& Gf_qmc,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& Gf_save);

    void output(const int char_step, const int DMFT_step, 
          const double mu, DMFT::input_info& in, 
          DFT_output::atoms_info& atom, DFT_output::KS_bands& band,
          std::vector<std::vector<DFT_plus_DMFT::Matrix>>& Eimp,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& Sigma_in,
          std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& Weiss);

    //========================interface=========================
    std::vector<std::vector<DFT_plus_DMFT::Matrix>>&
          hopping_term(){return hopping_matrix;}
    
    //Interfaces
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
        hybridization_func_tau(){return hyb_tau;}

    private:
    //hyb_tau[ineq][is][i_tau][m_index];
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>> hyb_tau;

    //hopping_term[ineq][is][m_index]
    std::vector<std::vector<DFT_plus_DMFT::Matrix>> hopping_matrix;

  };
}
