#pragma once
#include "../matrix_types.h"
#include <array>
#include <vector>

namespace DFT_plus_DMFT
{
    class solver;
    class Charge_SCF_abacus
    {
    public:
        void init();
        bool update(solver& task, int charge_step);
    private:
        std::vector<std::array<int,3>> R4dm;
        std::vector<RealMatrix> DMR_in;
        std::vector<double> rho_in;
        double dv;
    };
}
