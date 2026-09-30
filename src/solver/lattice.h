#pragma once

#include <complex>
#include <vector>

namespace DFT_plus_DMFT
{
namespace lattice
{
    std::vector<std::vector<std::complex<double>>> self_energy(
        int n_band_window, int n_omega,
        const std::vector<int>& n_orbitals,
        const std::vector<std::vector<std::complex<double>>>& P,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& Sigma_local);

    std::vector<std::complex<double>> green_function(
        const std::vector<double>& epsilon,
        const std::vector<std::complex<double>>& Sigma,
        std::complex<double> z, double mu);
}
}
