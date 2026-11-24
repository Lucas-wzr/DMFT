#include "overlap_matrix.h"
#include "../debug.h"
#include "../global_variables.h"

#include <iostream>
#include <cstdlib>

namespace DFT_output
{
  overlap_matrix::overlap_matrix(const std::string& DFT_solver, const std::string dir)
  : dft_solver(DFT_solver)
  {
    if(DFT_solver == "aims") this->ovlp_aims.read(dir);
    else if(DFT_solver == "abacus") this->ovlp_abacus.read(dir);
    else {
      std::cerr<< "Not supported DFT_solver" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }


  void overlap_matrix::evaluate_ovlp_k(const int ik, atoms_info& at_info)
  {
    if(this->dft_solver == "aims") this->ovlp_aims.evaluate_ovlp_k(ik, at_info);
    else if(this->dft_solver == "abacus") this->ovlp_abacus.evaluate_ovlp_k(ik, at_info);
    else {
      std::cerr<< "Not supported DFT_solver" << std::endl;
      std::exit(EXIT_FAILURE);
    }

  }

  DFT_plus_DMFT::Matrix& overlap_matrix::overlap()
  {
    if(this->dft_solver == "aims") return this->ovlp_aims.ovlp_mat();
    if(this->dft_solver == "abacus") return this->ovlp_abacus.ovlp_mat();
    std::cerr<< "Not supported DFT_solver" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  DFT_plus_DMFT::Matrix& overlap_matrix::overlap_localorb()
  {
    if(this->dft_solver == "aims") return this->ovlp_aims.local_ovlp();
    if(this->dft_solver == "abacus") return this->ovlp_abacus.local_ovlp();
    std::cerr<< "Not supported DFT_solver" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  void overlap_matrix::out()
  {
    if(this->dft_solver == "aims") this->ovlp_aims.out();
    else if(this->dft_solver != "abacus") {
      std::cerr<< "Not supported DFT_solver" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }


}
