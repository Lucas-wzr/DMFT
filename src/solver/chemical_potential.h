#pragma once

#include "../para/KS_bands.h"
#include "../para/correlated_atoms.h"
#include "self_energy.h"
#include "../para/input.h"
#include "projector.h"
#include "Hilbert_space.h"

#include <Eigen/Core>
#include <complex>
#include <vector>

namespace DFT_plus_DMFT
{
  class chemical_potential
  {
    public:
    chemical_potential()
     : DFT_mu(0.0),
       sigma_corrected_mu(0.0),
       local_ele_num(0.0),
       mu_initialized(false)
    {}

    ~chemical_potential(){};
    
    void update_chemical_potential(
         const int axis_flag,
         DFT_output::KS_bands& band, 
         DFT_output::atoms_info& atom, 
         DFT_plus_DMFT::projector& proj,
         DMFT::self_energy& sigma,
         DMFT::input_info& in,
         DFT_plus_DMFT::Hilbert_space& space);

    void evaluate_mu_bisection_imag(
         DFT_output::KS_bands& band, 
         DFT_output::atoms_info& atom, 
         DFT_plus_DMFT::projector& proj,
         DMFT::self_energy& sigma,
         DMFT::input_info& in, 
         DFT_plus_DMFT::Hilbert_space& space);
    
    double evaluate_electrons_number_imag(
         DFT_output::KS_bands& band,
         DFT_plus_DMFT::Hilbert_space& space,
         DMFT::self_energy& sigma,
         DFT_output::atoms_info& atom,
         DFT_plus_DMFT::projector& proj,
         const double beta, const int mag, const double mu);

    void evaluate_mu_bisection_imag_DFT(
         DFT_output::KS_bands& band, 
         DFT_output::atoms_info& atom, 
         DMFT::input_info& in, 
         DFT_plus_DMFT::Hilbert_space& space);
    
    double evaluate_electrons_number_imag_DFT(
         DFT_output::KS_bands& band,
         DFT_plus_DMFT::Hilbert_space& space,
         const double beta, const double mu);

    //Read chemical potential from DMFT_running.log; Used for spectrum function calculation
    void read_chemical_potential();       

    // --- New: build epsilon_infty using fitted static self-energy  ---
void build_epsilon_infty_from_fitted_static(
    DFT_output::KS_bands& band,
    DFT_output::atoms_info& atom,
    DFT_plus_DMFT::projector& proj,
    DMFT::self_energy& sigma,
    DMFT::input_info& in,
    DFT_plus_DMFT::Hilbert_space& space,
    int n_tail_fit);

// --- New: electrons number on imaginary axis using fitted Sigma_stat for tail correction ---
double evaluate_electrons_number_imag_fit(
    DFT_output::KS_bands& band,
    DFT_plus_DMFT::Hilbert_space& space,
    DMFT::self_energy& sigma,
    DFT_output::atoms_info& atom,
    DFT_plus_DMFT::projector& proj,
    const double beta, const int mag, const double mu,
    int n_tail_fit);

    
    //interfaces
    double mu_corrected(){return sigma_corrected_mu;}
    double mu_DFT(){return DFT_mu;}
    const std::vector<std::vector<std::vector<double>>>& fik(){return fik_wind;}

    private:

    double DFT_mu;
    double sigma_corrected_mu;
    double local_ele_num;
    bool mu_initialized;

    std::vector<std::vector<std::vector<double>>> fik_wind;       //fik_wind[is][ik][iband]

    std::vector<std::vector<Eigen::VectorXd>> epsilon_infty;  //epsilon_infty[is][ik][iband]

  };
}
