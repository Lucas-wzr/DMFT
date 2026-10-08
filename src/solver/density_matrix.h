#pragma once

#include <complex>
#include <array>
#include <string>
#include <vector>
#include "../matrix_types.h"

namespace DFT_plus_DMFT
{
namespace density_matrix
{
    using DFT_plus_DMFT::Matrix;
    using DFT_plus_DMFT::RealMatrix;

    // Collinear, real ABACUS DMR: spin_factor * sum_k w_k Re[exp(-2*pi*i*k.R) DMK(k)].
    // DMK(k)=C DM_full C^dagger is unweighted; k is fractional reciprocal, R integer lattice.
    // Weights sum to one per spin. spin_factor=2 for nspin=1, 1 per nspin=2 channel.
    // Supply a full mesh, or time-reversal pairs represented by their combined weights
    // only when DMK(-k)=DMK(k)*. Other symmetry-reduced meshes must be expanded first.
    std::vector<RealMatrix> density_matrix_R(
        const std::vector<Matrix>& DMK,
        const std::vector<std::array<double, 3>>& k,
        const std::vector<double>& weights,
        const std::vector<std::array<int, 3>>& R, double spin_factor);

    // One spin channel, global NAO order; caller chooses dmrs1_nao.csr / dmrs2_nao.csr.
    // Compact header bypasses the current ABACUS verbose-header reader bug.
    void write_density_matrix_R(
        const std::string& filename,
        const std::vector<std::array<int, 3>>& R,
        const std::vector<RealMatrix>& DMR, double threshold = 1e-10);

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

    // Full KS band matrix: window block is DM_window, exterior occupations
    // are Fermi values, and window-exterior cross terms are zero.
    Matrix density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu);

    Matrix density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu);

    // C(mu,n), window[i] is the full KS band index of window band i.
    // DMK = C DM_full C^dagger.
    Matrix density_matrix_NAO(
        const Eigen::Ref<const Matrix>& C,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu);

    // Total band occupation at this k/spin, including all window-exterior bands.
    // The caller sums with k weights and, for nonmagnetic non-SOC, a factor 2.
    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu);

    // Interfaces for flattened row-major vector matrices.
    std::vector<std::vector<double>> density_matrix_R(
        const std::vector<std::vector<std::complex<double>>>& DMK, int nbasis,
        const std::vector<std::array<double, 3>>& k,
        const std::vector<double>& weights,
        const std::vector<std::array<int, 3>>& R, double spin_factor);

    void write_density_matrix_R(
        const std::string& filename,
        const std::vector<std::array<int, 3>>& R,
        const std::vector<std::vector<double>>& DMR, int nbasis,
        double threshold = 1e-10);

    std::vector<std::complex<double>> density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& sigma,
        double beta, double mu, int n_tail_fit = 30);

    std::vector<std::complex<double>> density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu);

    std::vector<std::complex<double>> density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu);

    std::vector<std::complex<double>> density_matrix_NAO(
        const std::vector<std::complex<double>>& C, int nbasis,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu);

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu);
}
}
