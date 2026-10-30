#pragma once
#include "../matrix_types.h"

#include "../para/correlated_atoms.h"
#include "self_energy_real_aixs.h"
#include "self_energy_imaginary_aixs.h"
#include "double_counting.h"
#include "projector.h"

#include <vector>

#include <complex>

namespace DMFT
{
  class self_energy
  {
    public:
    self_energy();
    ~self_energy();

    self_energy_real_aixs sigma_real;
    self_energy_imaginary_aixs sigma_imag;
    double_counting dc;

    bool read_sigma_save(const int impurity_solver, const bool SOC,
                         const int nspin, DFT_output::atoms_info& atom);

    void initial_guess(const int axis_flag, const bool SOC,
                       const int nspin, DFT_output::atoms_info& atom);

    void subtract_double_counting(const int axis_flag);  //0: imaginary axis, 1:real axis

    void evalute_lattice_sigma(
        const int axis_flag, const int mag,
        const int ispin, const std::vector<int>& wbands,
        DFT_output::atoms_info& atom,
        const std::vector<std::vector<DFT_plus_DMFT::Matrix>>& projector,
        std::vector<DFT_plus_DMFT::Matrix>& Sigma);

    void evalute_lattice_sigma_infty(
        const int axis_flag, const int mag,
        const int ispin, const std::vector<int>& wbands,
        DFT_output::atoms_info& atom,
        const std::vector<std::vector<DFT_plus_DMFT::Matrix>>& projector,
        DFT_plus_DMFT::Matrix& Sigma_infty);

    // void embeding_self_energy(const int impurity_solver, )
    // Fit static part Σ_stat ≈ Re Σ(i∞) from Matsubara data.
    // lattice_Sigma: one dim x dim matrix per frequency.
    // freq: [nomega], Matsubara frequencies ω_n > 0 (double).
    // dim: matrix dimension.
    // n_tail_fit: number of tail points used for fit (e.g. 40~80).
    // out_sigma_stat: dim x dim, Hermitianized, Im part set to 0 for stability.
    void fit_sigma_static_from_tail(
        const std::vector<DFT_plus_DMFT::Matrix>& lattice_Sigma,
        const std::vector<double>& freq,
        int dim,
        int n_tail_fit,
        DFT_plus_DMFT::Matrix& out_sigma_stat);

    void out();    //test whether reading worked corrected

    // =================================================
    //             interface
    // =================================================
    int nomega(const int axis_flag); //0: imaginary axis, 1:real axis

    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
              sigma_new(const int axis_flag); //0: imaginary axis, 1:real axis

    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
              sigma_save(const int axis_flag); //0: imaginary axis, 1:real axis

    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
              correlated_sigma(const int axis_flag); //0: imaginary axis, 1:real axis

  };
}
