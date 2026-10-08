#include "density_matrix.h"
#include "../constants.h"
#include "lattice.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iomanip>

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

    std::vector<RealMatrix> density_matrix_R(
        const std::vector<Matrix>& DMK,
        const std::vector<std::array<double, 3>>& k,
        const std::vector<double>& weights,
        const std::vector<std::array<int, 3>>& R, double spin_factor)
    {
        if(DMK.empty() || DMK.size()!=k.size() || k.size()!=weights.size())
            throw std::invalid_argument("density_matrix_R: inconsistent k arrays");
        const int nbasis = DMK.front().rows();
        auto DMR = std::vector<RealMatrix>(R.size(), RealMatrix::Zero(nbasis, nbasis));
        for(int ik=0; ik<int(k.size()); ik++){
            for(int ir=0; ir<int(R.size()); ir++){
                double kR = 0.0;
                for(int axis=0; axis<3; axis++) kR += k[ik][axis]*R[ir][axis];
                const auto phase = std::exp(std::complex<double>(0.0, -2.0*GLC::PI*kR));
                DMR[ir] += spin_factor*weights[ik]*(phase*DMK[ik]).real();
            }
        }
        return DMR;
    }

    void write_density_matrix_R(
        const std::string& filename,
        const std::vector<std::array<int, 3>>& R,
        const std::vector<RealMatrix>& DMR, double threshold)
    {
        const int nbasis = DMR.front().rows();
        std::ofstream output(filename);
        output << "STEP: 1\nMatrix Dimension of density matrix: " << nbasis
               << "\nMatrix number of density matrix: " << R.size() << "\n";
        output << std::scientific << std::setprecision(17);
        for(int ir=0; ir<int(R.size()); ir++){
            std::vector<double> values;
            std::vector<int> columns;
            auto row_pointer = std::vector<int>(nbasis+1, 0);
            for(int row=0; row<nbasis; row++){
                for(int col=0; col<nbasis; col++){
                    const double value = DMR[ir](row,col);
                    if(std::abs(value)>threshold){
                        values.push_back(value);
                        columns.push_back(col);
                    }
                }
                row_pointer[row+1] = values.size();
            }
            output << R[ir][0] << " " << R[ir][1] << " " << R[ir][2] << " " << values.size() << "\n";
            for(double value : values) output << value << " ";
            output << "\n";
            for(int col : columns) output << col << " ";
            output << "\n";
            for(int offset : row_pointer) output << offset << " ";
            output << "\n";
        }
        output.close();
        if(!output) throw std::runtime_error("Failed writing density matrix: "+filename);
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
        Matrix DM_window = eig.eigenvectors() * f.asDiagonal() * eig.eigenvectors().adjoint();

        auto z = std::vector<std::complex<double>>(n_omega);
        for(int iw=0; iw<n_omega; iw++){
            z[iw] = std::complex<double>(0.0, (2*iw+1)*GLC::PI/beta);
        }
        const auto G = lattice::green_function(epsilon, sigma, z, mu);
        const auto G0 = lattice::green_function(epsilon, std::vector<Matrix>(n_omega, Sigma_static), z, mu);
        for(int iw=0; iw<n_omega; iw++){
            // Negative frequencies contribute G(iw)^dagger
            const Matrix delta_G = G[iw] - G0[iw];
            DM_window += (delta_G + delta_G.adjoint()) / beta;
        }
        return DM_window;
    }

    Matrix density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu)
    {
        const int nbands = epsilon.size();
        const int n_window = window.size();
        auto DM_full = Matrix(Matrix::Zero(nbands, nbands));
        for(int b=0; b<nbands; b++)
            DM_full(b,b) = fermi(epsilon[b], beta, mu);
        for(int i=0; i<n_window; i++)
            for(int j=0; j<n_window; j++)
                DM_full(window[i],window[j]) = DM_window(i,j);
        return DM_full;
    }

    Matrix density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu)
    {
        return density_matrix_full(epsilon, window, DM_window, beta, mu);
    }

    Matrix density_matrix_NAO(
        const Eigen::Ref<const Matrix>& C,
        const std::vector<double>& epsilon,
        const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu)
    {
        const auto DM_full = density_matrix(epsilon, window, DM_window, beta, mu);
        return C * DM_full * C.adjoint();
    }

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const Eigen::Ref<const Matrix>& DM_window,
        double beta, double mu)
    {
        return density_matrix(epsilon, window, DM_window, beta, mu).trace().real();
    }

    // Interfaces for flattened row-major vector matrices.
    std::vector<std::vector<double>> density_matrix_R(
        const std::vector<std::vector<std::complex<double>>>& DMK, int nbasis,
        const std::vector<std::array<double, 3>>& k,
        const std::vector<double>& weights,
        const std::vector<std::array<int, 3>>& R, double spin_factor)
    {
        auto matrices = std::vector<Matrix>(DMK.size());
        for(int ik=0; ik<int(DMK.size()); ik++){
            if(nbasis<=0 || DMK[ik].size()!=std::size_t(nbasis)*nbasis)
                throw std::invalid_argument("density_matrix_R: invalid vector matrix size");
            matrices[ik] = Eigen::Map<const Matrix>(DMK[ik].data(), nbasis, nbasis);
        }
        const auto DMR = density_matrix_R(matrices, k, weights, R, spin_factor);
        auto result = std::vector<std::vector<double>>(R.size());
        for(int ir=0; ir<int(R.size()); ir++)
            result[ir].assign(DMR[ir].data(), DMR[ir].data()+DMR[ir].size());
        return result;
    }

    void write_density_matrix_R(
        const std::string& filename,
        const std::vector<std::array<int, 3>>& R,
        const std::vector<std::vector<double>>& DMR, int nbasis, double threshold)
    {
        auto matrices = std::vector<RealMatrix>(DMR.size());
        for(int ir=0; ir<int(DMR.size()); ir++){
            if(nbasis<=0 || DMR[ir].size()!=std::size_t(nbasis)*nbasis)
                throw std::invalid_argument("write_density_matrix_R: invalid vector matrix size");
            matrices[ir] = Eigen::Map<const RealMatrix>(DMR[ir].data(), nbasis, nbasis);
        }
        write_density_matrix_R(filename, R, matrices, threshold);
    }

    std::vector<std::complex<double>> density_matrix_window(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& sigma,
        double beta, double mu, int n_tail_fit)
    {
        const int n = epsilon.size();
        auto Sigma = std::vector<Matrix>(sigma.size());
        for(int iw=0; iw<int(sigma.size()); iw++)
            Sigma[iw] = Eigen::Map<const Matrix>(sigma[iw].data(), n, n);
        const auto DM_window = density_matrix_window(epsilon, Sigma, beta, mu, n_tail_fit);
        return std::vector<std::complex<double>>(DM_window.data(), DM_window.data()+DM_window.size());
    }

    std::vector<std::complex<double>> density_matrix_full(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu)
    {
        const int n_window = window.size();
        const auto DM_window_matrix = Eigen::Map<const Matrix>(DM_window.data(), n_window, n_window);
        const auto DM_full = density_matrix_full(epsilon, window, DM_window_matrix, beta, mu);
        return std::vector<std::complex<double>>(DM_full.data(), DM_full.data()+DM_full.size());
    }

    std::vector<std::complex<double>> density_matrix(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu)
    {
        return density_matrix_full(epsilon, window, DM_window, beta, mu);
    }

    std::vector<std::complex<double>> density_matrix_NAO(
        const std::vector<std::complex<double>>& C, int nbasis,
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu)
    {
        const int nbands = epsilon.size();
        const int n = window.size();
        if(nbasis <= 0 || int(C.size()) != nbasis*nbands || int(DM_window.size()) != n*n)
            throw std::invalid_argument("density_matrix::density_matrix_NAO: invalid dimensions");
        const auto C_matrix = Eigen::Map<const Matrix>(C.data(), nbasis, nbands);
        const auto DM_window_matrix = Eigen::Map<const Matrix>(DM_window.data(), n, n);
        const auto DMK = density_matrix_NAO(C_matrix, epsilon, window, DM_window_matrix, beta, mu);
        return std::vector<std::complex<double>>(DMK.data(), DMK.data()+DMK.size());
    }

    double electron_number(
        const std::vector<double>& epsilon, const std::vector<int>& window,
        const std::vector<std::complex<double>>& DM_window,
        double beta, double mu)
    {
        const int n = window.size();
        if(int(DM_window.size()) != n*n)
            throw std::invalid_argument("density_matrix::electron_number: invalid dimensions");
        const auto DM_window_matrix = Eigen::Map<const Matrix>(DM_window.data(), n, n);
        return electron_number(epsilon, window, DM_window_matrix, beta, mu);
    }
}
}
