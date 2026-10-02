#include "density_matrix.h"
#include "../constants.h"
#include "lattice.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace DFT_plus_DMFT
{
namespace density_matrix
{
    namespace
    {
        double fermi(double energy, double beta, double mu)
        {
            const double x = beta * (energy - mu);
            const double e = std::exp(-std::abs(x));
            return x >= 0.0 ? e / (1.0 + e) : 1.0 / (1.0 + e);
        }
    }

    Matrix density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<Matrix>& sigma,
        double beta, double mu, int n_tail_fit)
    {
        const int n_band_window = epsilon.size();
        const int n_omega = sigma.size();

        const int nfit = std::min(n_tail_fit, n_omega);
        std::vector<double> x(nfit);
        double xmean = 0.0, variance = 0.0;
        // Fit the Hermitian part of Sigma = Sigma_static + O(omega^-2).
        // Scaling x by the last frequency avoids a tiny regression denominator.
        for(int t=0; t<nfit; t++){
            x[t] = std::pow(double(2*n_omega-1) / (2*(n_omega-nfit+t)+1), 2);
            xmean += x[t] / nfit;
        }
        for (double value : x) {
            variance += (value - xmean) * (value - xmean);
        }

        Matrix ymean = Matrix::Zero(n_band_window, n_band_window);
        Matrix covariance = Matrix::Zero(n_band_window, n_band_window);
        for(int t=0; t<nfit; t++){
            const auto& Sigma = sigma[n_omega-nfit+t];
            const Matrix y = 0.5 * (Sigma + Sigma.adjoint());
            ymean += y / double(nfit);
            covariance += (x[t]-xmean) * y;
        }
        Matrix slope = covariance / variance;
        Matrix Sigma_static = ymean - xmean * slope;
        double residual_squared = 0.0;
        for(int t=0; t<nfit; t++){
            const auto& Sigma = sigma[n_omega-nfit+t];
            const Matrix y = 0.5 * (Sigma + Sigma.adjoint());
            const Matrix residual = y - Sigma_static - x[t] * slope;
            residual_squared += residual.squaredNorm();
        }
        // Average over fitted frequencies and matrix elements.
        const double fit_error_eV = std::sqrt(residual_squared
            / (double(nfit) * n_band_window * n_band_window)) * GLC::Hartree_to_eV;
        if(!std::isfinite(residual_squared) || !(fit_error_eV < 1.e-5))
            throw std::runtime_error("density_matrix::density_matrix_window: RMS tail fit residual must be < 1e-5 eV");
        
        Matrix h = Sigma_static;
        for(int i=0; i<n_band_window; i++){
            h(i,i) += epsilon[i];
        }

        // h = U E U^dagger; retain U to return f(h-mu) in the KS basis.
        Eigen::SelfAdjointEigenSolver<Matrix> eig(h);
        Eigen::VectorXd f(n_band_window);
        for(int i=0; i<n_band_window; i++) {
            f(i) = fermi(eig.eigenvalues()(i), beta, mu);
        }
        Matrix rho_window = eig.eigenvectors() * f.asDiagonal() * eig.eigenvectors().adjoint();

        auto z = std::vector<std::complex<double>>(n_omega);
        for(int iw=0; iw<n_omega; iw++){
            z[iw] = std::complex<double>(0.0, (2*iw+1)*GLC::PI/beta);
        }
        const auto G = lattice::green_function(epsilon, sigma, z, mu);
        const auto G0 = lattice::green_function(epsilon, std::vector<Matrix>(n_omega, Sigma_static), z, mu);
        for(int iw=0; iw<n_omega; iw++){
            // Negative frequencies contribute G(iw)^dagger
            const Matrix delta_G = G[iw] - G0[iw];
            rho_window += (delta_G + delta_G.adjoint()) / beta;
        }
        return rho_window;
    }

    Matrix density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu)
    {
        const int nbands = epsilon.size();
        const int n_window = window.size();
        auto rho_full = Matrix(Matrix::Zero(nbands, nbands));
        for(int b=0; b<nbands; b++)
            rho_full(b,b) = fermi(epsilon[b], beta, mu);
        for(int i=0; i<n_window; i++)
            for(int j=0; j<n_window; j++)
                rho_full(window[i],window[j]) = rho_window(i,j);
        return rho_full;
    }

    Matrix density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu)
    {
        return density_matrix_full(epsilon, window, rho_window, beta, mu);
    }

    Matrix density_matrix_NAO(
        const Eigen::Ref<const Matrix>& C,
        const std::vector<double>& epsilon,
        const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu)
    {
        const auto rho_full = density_matrix(epsilon, window, rho_window, beta, mu);
        return C * rho_full * C.adjoint();
    }

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& rho_window,
        double beta, double mu)
    {
        return density_matrix(epsilon, window, rho_window, beta, mu).trace().real();
    }

    // Interfaces for flattened row-major vector matrices.
    std::vector<std::complex<double>> density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& sigma,
        double beta, double mu, int n_tail_fit)
    {
        const int n = epsilon.size();
        auto Sigma = std::vector<Matrix>(sigma.size());
        for(int iw=0; iw<int(sigma.size()); iw++)
            Sigma[iw] = Eigen::Map<const Matrix>(sigma[iw].data(), n, n);
        const auto rho_window = density_matrix_window(epsilon, Sigma, beta, mu, n_tail_fit);
        return std::vector<std::complex<double>>(rho_window.data(), rho_window.data()+rho_window.size());
    }

    std::vector<std::complex<double>> density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu)
    {
        const int n_window = window.size();
        const auto rho_window_matrix = Eigen::Map<const Matrix>(rho_window.data(), n_window, n_window);
        const auto rho_full = density_matrix_full(epsilon, window, rho_window_matrix, beta, mu);
        return std::vector<std::complex<double>>(rho_full.data(), rho_full.data()+rho_full.size());
    }

    std::vector<std::complex<double>> density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu)
    {
        return density_matrix_full(epsilon, window, rho_window, beta, mu);
    }

    std::vector<std::complex<double>> density_matrix_NAO(
        const std::vector<std::complex<double>>& C, int nbasis,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu)
    {
        const int nbands = epsilon.size();
        const int n = window.size();
        if(nbasis <= 0 || int(C.size()) != nbasis*nbands || int(rho_window.size()) != n*n)
            throw std::invalid_argument("density_matrix::density_matrix_NAO: invalid dimensions");
        const auto C_matrix = Eigen::Map<const Matrix>(C.data(), nbasis, nbands);
        const auto rho_window_matrix = Eigen::Map<const Matrix>(rho_window.data(), n, n);
        const auto D = density_matrix_NAO(C_matrix, epsilon, window, rho_window_matrix, beta, mu);
        return std::vector<std::complex<double>>(D.data(), D.data()+D.size());
    }

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& rho_window,
        double beta, double mu)
    {
        const int n = window.size();
        if(int(rho_window.size()) != n*n)
            throw std::invalid_argument("density_matrix::electron_number: invalid dimensions");
        const auto rho_window_matrix = Eigen::Map<const Matrix>(rho_window.data(), n, n);
        return electron_number(epsilon, window, rho_window_matrix, beta, mu);
    }
}
}
