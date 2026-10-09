#include "coulomb_tensor.h"

#include "../debug.h"
#include "../constants.h"
#include "rutgers_atomic.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace DMFT
{
  void coulomb_tensor::update_coulomb_tensor(
    const int type, DFT_output::atoms_info& atom)
  {
    try
    {
      if(type==1 || type==2)
      {
        this->Kana_coulomb.evaluate_coulomb_tensor(atom, this->U_matrix, type==2);
        return;
      }
      this->U_matrix.resize(atom.inequ_atoms());
      for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
      {
        const int iatom = atom.ineq_iatom(ineq);
        const int M = atom.iatom_norb()[iatom];
        if(type==3)
        {
          const std::string file = "Uijkl_impurity" + std::to_string(ineq) + ".dat";
          std::ifstream input(file.c_str());
          if(!input) throw std::runtime_error("Cannot open " + file);
          this->U_matrix[ineq] = rutgers_atomic::read_tensor(input, M, GLC::Hartree_to_eV);
        }
        else throw std::runtime_error("Unsupported interaction type");
      }
    }
    catch(const std::exception& error)
    {
      std::cerr << "Coulomb tensor: " << error.what() << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  void coulomb_tensor::out_coulomb_tensor(
            const int type,
            const int char_step,
            const int DMFT_step,
            const int impurity_solver, 
            DFT_output::atoms_info& atom,
            DFT_output::KS_bands& band)
  {
    switch(type)
    {
      case 1:
        if(impurity_solver==1)
          this->Kana_coulomb.out_ALPS_CTHYB(char_step, DMFT_step, atom, band);
        else if(impurity_solver==2)
          this->Kana_coulomb.out_ALPS_CTHYB_SEGMENT(char_step, DMFT_step, atom, band);
        break;
      default:
        break;
    }
  }
  
}
