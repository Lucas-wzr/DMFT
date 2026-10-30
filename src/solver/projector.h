#pragma once

#include "../para/overlap_matrix.h"
#include "../para/KS_eigenvectors.h"
#include "../para/correlated_atoms.h"
#include "../para/KS_bands.h"
#include "Hilbert_space.h"
#include "../matrix_types.h"

#include <complex>
#include <vector>


//TEST
#include "../para/KS_bands.h"
namespace DFT_plus_DMFT
{
  class projector
  {
    public:
    projector(): n_basis(0) {};
    ~projector(){};

    void evaluate_projector(
        const int projector_type,
        const int DFT_solver,
        DFT_output::KS_bands& band,
        DFT_plus_DMFT::Hilbert_space& space,
        DFT_output::atoms_info& atom,
        const double CWF_ep0,
        const double CWF_ep1,
        const double CWF_T,
        const double projection_min_weight = 0.9);

    void evalute_projector_k(
          DFT_output::overlap_matrix& ovlp,
          DFT_output::KS_eigenvectors& wfc,
          DFT_output::atoms_info& atom,
          const int ik,
          const int ik_count);


    //===========================
    //   interface
    //===========================
    const std::vector<std::vector<Matrix>>&
          proj_access(const int ik) const {return projector_mat[ik];}
    
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>>&
          Hloc(){return TB_Hk;}

    const int& nbasis(){return n_basis;} 
    
    private:
    int n_basis;

    //projector_mat[ik][iatom][is]: nbands x m_tot
    std::vector<std::vector<std::vector<Matrix>>> projector_mat;
  
    //unitary_trans[ik][iatom][is]: m_tot x m_tot
    std::vector<std::vector<std::vector<Matrix>>> unitary_trans;
    //Tight binding Haimiltonian:TB_Hk[ik][iatom][nspin][m_tot];
    std::vector<std::vector<std::vector<std::vector<std::complex<double>>>>> TB_Hk;
    
  };
}
