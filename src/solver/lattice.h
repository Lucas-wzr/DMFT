#pragma once

#include <complex>
#include <vector>
#include "../matrix_types.h"

namespace DFT_plus_DMFT
{
namespace lattice
{
    std::vector<Matrix> self_energy(
        int n_band_window, int n_omega,
        const std::vector<Matrix>& P,
        const std::vector<std::vector<Matrix>>& Sigma_local);

    // Sigma[iw] corresponds to z[iw]; energies and frequencies are in Hartree.
    std::vector<Matrix> green_function(
        const std::vector<double>& epsilon,
        const std::vector<Matrix>& Sigma,
        const std::vector<std::complex<double>>& z, double mu);

    std::vector<std::vector<std::complex<double>>> self_energy(
        int n_band_window, int n_omega,
        const std::vector<int>& n_orbitals,
        const std::vector<std::vector<std::complex<double>>>& P,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& Sigma_local);

    std::vector<std::vector<std::complex<double>>> green_function(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& Sigma,
        const std::vector<std::complex<double>>& z, double mu);
}
}
