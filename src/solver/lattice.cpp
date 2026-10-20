#include "lattice.h"
#include <Eigen/Dense>

#include <stdexcept>

namespace DFT_plus_DMFT
{
namespace lattice
{
    std::vector<Matrix> self_energy(
        int n_band_window, int n_omega,
        const std::vector<Matrix>& P,
        const std::vector<std::vector<Matrix>>& Sigma_local)
    {
        auto Sigma = std::vector<Matrix>(n_omega, Matrix::Zero(n_band_window, n_band_window));
        const int natom = int(P.size());
        for(int iatom=0; iatom<natom; iatom++)
            for(int iw=0; iw<n_omega; iw++)
                Sigma[iw] += P[iatom] * Sigma_local[iatom][iw] * P[iatom].adjoint();
        return Sigma;
    }

    std::vector<Matrix> green_function(
        const std::vector<double>& epsilon,
        const std::vector<Matrix>& Sigma,
        const std::vector<std::complex<double>>& z, double mu)
    {
        const int n_band_window = epsilon.size();
        const int n_omega = z.size();
        auto G = std::vector<Matrix>(n_omega);
        for(int iw=0; iw<n_omega; iw++){
            auto inverse_G = Matrix(-Sigma[iw]);
            for(int i=0; i<n_band_window; i++)
                inverse_G(i,i) += z[iw]+mu-epsilon[i];
            G[iw] = inverse_G.inverse();
        }
        return G;
    }

    // interface for old data type
    std::vector<std::vector<std::complex<double>>> self_energy(
        int n_band_window, int n_omega,
        const std::vector<int>& n_orbitals,
        const std::vector<std::vector<std::complex<double>>>& P,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& Sigma_local)
    {
        const int natom = int(n_orbitals.size());
        auto P_matrix = std::vector<Matrix>(natom);
        auto Sigma_local_matrix = std::vector<std::vector<Matrix>>(natom, std::vector<Matrix>(n_omega));
        for(int iatom=0; iatom<natom; iatom++){
            const int norb = n_orbitals[iatom];
            P_matrix[iatom] = Eigen::Map<const Matrix>(P[iatom].data(), n_band_window, norb);
            for(int iw=0; iw<n_omega; iw++)
                Sigma_local_matrix[iatom][iw] = Eigen::Map<const Matrix>(Sigma_local[iatom][iw].data(), norb, norb);
        }
        const auto Sigma_matrix = self_energy(n_band_window, n_omega, P_matrix, Sigma_local_matrix);
        auto Sigma = std::vector<std::vector<std::complex<double>>>(n_omega);
        for(int iw=0; iw<n_omega; iw++)
            Sigma[iw].assign(Sigma_matrix[iw].data(), Sigma_matrix[iw].data()+Sigma_matrix[iw].size());
        return Sigma;
    }

    std::vector<std::vector<std::complex<double>>> green_function(
        const std::vector<double>& epsilon,
        const std::vector<std::vector<std::complex<double>>>& Sigma,
        const std::vector<std::complex<double>>& z, double mu)
    {
        const int n_band_window = epsilon.size();
        const int n_omega = z.size();
        auto Sigma_matrix = std::vector<Matrix>(n_omega);
        for(int iw=0; iw<n_omega; iw++)
            Sigma_matrix[iw] = Eigen::Map<const Matrix>(Sigma[iw].data(), n_band_window, n_band_window);
        const auto G_matrix = green_function(epsilon, Sigma_matrix, z, mu);
        auto G = std::vector<std::vector<std::complex<double>>>(n_omega);
        for(int iw=0; iw<n_omega; iw++)
            G[iw].assign(G_matrix[iw].data(), G_matrix[iw].data()+G_matrix[iw].size());
        return G;
    }
}
}
