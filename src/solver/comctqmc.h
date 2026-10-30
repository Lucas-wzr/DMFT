#pragma once
#include "../matrix_types.h"
#include "coulomb_tensor.h"
#include "../para/input.h"
#include <complex>

namespace DMFT
{
    class ComCTQMC
    {
        public:
        // Hartree inputs; beta in Ha^-1. Spin orbitals: all up, then all down.
        // directory must exist; files contain static interactions only.
        void write_input(const std::string& directory, double beta, double mu,
            const std::vector<double>& levels,
            const std::vector<std::vector<std::complex<double>>>& delta,
            const interaction_tensor& U, long long thermalisation_steps, long long measurement_steps);

        void output(int charge_step, int dmft_step, double mu, input_info& in,
            DFT_output::atoms_info& atom, DFT_output::KS_bands& band,
            const std::vector<std::vector<DFT_plus_DMFT::Matrix>>& Eimp,
            const std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& hyb,
            coulomb_tensor& Umat);

        void run(const std::string& directory, const std::string& executable,
            const std::string& evalsim, const std::string& launcher, int nproc);

        // Returned G and Sigma use Hartree units, indexed by spin orbital and frequency.
        void read_functions(const std::string& directory, double beta, int L, int N,
            std::vector<std::vector<std::complex<double>>>& G,
            std::vector<std::vector<std::complex<double>>>& S);

        void impurities_solving(int charge_step, int dmft_step,
            DFT_output::atoms_info& atom);

        void read_self_energy(int charge_step, int dmft_step, DFT_output::KS_bands& band,
            input_info& in, DFT_output::atoms_info& atom,
            std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>& S);

        private:
        std::string executable="CTQMC", evalsim="EVALSIM", launcher="comctqmc-launch";
        int nproc=0;
    };
}
