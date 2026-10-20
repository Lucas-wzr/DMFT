#pragma once
#include "coulomb_tensor.h"
#include <istream>

namespace DMFT
{
  namespace slater_condon
  {
    // F={F0,F2,...,F2l}; same energy unit in input and output.
    // Real tesseral harmonics labelled m=-l,...,l, Condon--Shortley phases.
    interaction_tensor evaluate(int l, const std::vector<double>& F,
                                   const std::vector<int>& orbitals);
    std::vector<double> read_integrals(std::istream& input, int l, double energy_unit);
    // U is the shell average F0, not the Kanamori intra-orbital repulsion.
    std::vector<double> from_U_J(int l, double U, double J,
                                double f4_f2=-1.0, double f6_f2=-1.0);
  }
}
