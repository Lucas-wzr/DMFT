#pragma once

#include <complex>
#include <vector>
#include <Eigen/Core>

namespace DFT_plus_DMFT
{
namespace density_matrix
{
    using Matrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // One k point and one spin, row-major matrices; no k weight or spin factor.
    // Energies, mu and sigma are in Hartree; beta in Hartree^-1.
    // sigma already includes the double-counting subtraction, on positive
    // fermionic Matsubara frequencies (2*n+1)*pi/beta, starting at n=0.
    // Fit the last 30 points by default; require Hermitian-part fit RMS
    // over fitted frequencies and all matrix elements < 1e-5 eV, otherwise throw.
    Matrix density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<Matrix>& sigma,
        double beta, double mu, int n_tail_fit = 30);

    // Full KS band matrix: window block is rho_window, exterior occupations
    // are Fermi values, and window-exterior cross terms are zero.
    Matrix density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu);

    Matrix density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu);

    // C(mu,n), window[i] is the full KS band index of window band i.
    // D = C rho_full C^dagger.
    Matrix density_matrix_NAO(
        const Eigen::Ref<const Matrix>& C,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu);

    // Total band occupation at this k/spin, including all window-exterior bands.
    // The caller sums with k weights and, for nonmagnetic non-SOC, a factor 2.
    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu);

    // Interfaces for flattened row-major vector matrices.
    std::vector<std::complex<double>> density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& sigma,
        double beta, double mu, int n_tail_fit = 30);

    std::vector<std::complex<double>> density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu);

    std::vector<std::complex<double>> density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu);

    std::vector<std::complex<double>> density_matrix_NAO(
        const std::vector<std::complex<double>>& C, int nbasis,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu);

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu);
}
}
