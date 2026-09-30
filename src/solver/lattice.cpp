#include "lattice.h"
#include <Eigen/Dense>

#include <stdexcept>

namespace DFT_plus_DMFT
{
namespace lattice
{
    using Matrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    std::vector<std::vector<std::complex<double>>> self_energy(
        int n_band_window, int n_omega,
        const std::vector<int>& n_orbitals,
        const std::vector<std::vector<std::complex<double>>>& P,
        const std::vector<std::vector<std::vector<std::complex<double>>>>& Sigma_local)
    {
        std::vector<std::vector<std::complex<double>>> Sigma(
            n_omega, std::vector<std::complex<double>>(n_band_window*n_band_window, 0.0));
        const int natom = int(n_orbitals.size());
        for(int iatom=0; iatom<natom; iatom++){
            const int norb = n_orbitals[iatom];
            const Eigen::Map<const Matrix> P_atom(P[iatom].data(), n_band_window, norb);
            for(int iw=0; iw<n_omega; iw++){
                const Eigen::Map<const Matrix> Sigma_atom(Sigma_local[iatom][iw].data(), norb, norb);
                Eigen::Map<Matrix> Sigma_k(Sigma[iw].data(), n_band_window, n_band_window);
                Sigma_k += P_atom * Sigma_atom * P_atom.adjoint();
            }
        }
        return Sigma;
    }

    std::vector<std::complex<double>> green_function(
        const std::vector<double>& epsilon,
        const std::vector<std::complex<double>>& Sigma,
        std::complex<double> z, double mu)
    {
        const int n_band_window = epsilon.size();
        const Eigen::Map<const Matrix> Sigma_k(Sigma.data(), n_band_window, n_band_window);
        Matrix inverse_G = -Sigma_k;
        for(int i=0; i<n_band_window; i++)
            inverse_G(i,i) += z+mu-epsilon[i];
        Matrix G = inverse_G.inverse();
        return std::vector<std::complex<double>>(G.data(), G.data()+G.size());
    }
}
}
