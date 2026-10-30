#include "self_energy.h"

#include "../debug.h"
#include "../timer.h"
#include "../constants.h"
#include "../global_variables.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include <Eigen/Dense>

using DFT_plus_DMFT::Matrix;

namespace DMFT
{
  self_energy::self_energy(){}

  self_energy::~self_energy(){}

  bool self_energy::read_sigma_save(const int impurity_solver, const bool SOC,
                    const int nspin, DFT_output::atoms_info& atom)
  {
    if(impurity_solver==1)
    {
      this->sigma_imag.read_sigma_save(SOC, nspin, atom);
    }
    return true;
  }

  void self_energy::initial_guess(const int axis_flag, const bool SOC,
            const int nspin, DFT_output::atoms_info& atom)
  {
    debug::codestamp("self_energy::initial_guess");

    const std::vector<int>& norb_sub = atom.iatom_norb();
    const std::complex<double> zero(0.0,0.0);
    
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
            Sigma_new = this->sigma_new(axis_flag);
    
    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
            Sigma_save = this->sigma_save(axis_flag);

    Sigma_new.resize(atom.inequ_atoms());
    for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
    {
      const int iatom = atom.ineq_iatom(ineq);
      const int m_tot=norb_sub[iatom];
      int nspin_tmp=nspin;
      // if(SOC) nspin_tmp=4;

      std::vector<std::vector<std::complex<double>>>& 
          Vdca = this->dc.Vdc()[ineq];

      const int n_omega=this->nomega(axis_flag);
        
      Sigma_new[ineq].resize(nspin_tmp);
 
      for(int is=0; is<nspin_tmp; is++)
      {
        Sigma_new[ineq][is].resize(n_omega);

        for(int i_omega=0; i_omega<n_omega; i_omega++)
        {
          Sigma_new[ineq][is][i_omega].setZero(m_tot, m_tot);

          for(int m_index=0; m_index<m_tot*m_tot; m_index++)
              Sigma_new[ineq][is][i_omega].data()[m_index] = Vdca[is][m_index];
        }//is
      }//i_omega
    }//iatom

    Sigma_save = Sigma_new;

    return;
  }

  void self_energy::evalute_lattice_sigma(
        const int axis_flag, const int mag,
        const int ispin, const std::vector<int>& wbands,
        DFT_output::atoms_info& atom,
        const std::vector<std::vector<Matrix>>& projector,
        std::vector<Matrix>& Sigma)
  {
    debug::codestamp("self_energy::evalute_lattice_sigma");

    const int nbands = wbands[ispin];
    const int omega_num = this->nomega(axis_flag);
    const auto& local_sigma = this->correlated_sigma(axis_flag);
    Sigma.assign(omega_num, Matrix::Zero(nbands, nbands));

    for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
    {
      const int iatom = atom.ineq_iatom(ineq);
      const int norb = atom.iatom_norb()[iatom];
      // Keep the representative-first summation order of the original code.
      std::vector<int> equivalent_atoms{iatom};
      for(int a=0; a<atom.total_atoms(); a++)
        if(a != iatom && atom.equ_atom(a)==ineq) equivalent_atoms.push_back(a);

      for(int iw=0; iw<omega_num; iw++)
        for(int a : equivalent_atoms)
        {
          int spin = ispin;
          if(a != iatom){
            const int flip = mag==1 && atom.magnetic(iatom)*atom.magnetic(a)==-1;
            spin = (ispin+flip)%2;
          }
          const Matrix& P = projector[a][ispin];
          const Matrix& local = local_sigma[ineq][spin][iw];
          // Preserve the diagonal-local-self-energy approximation.
          const Matrix weighted_P = P * local.diagonal().asDiagonal();
          Sigma[iw] += weighted_P * P.adjoint();
        }
    }
  }

  void self_energy::evalute_lattice_sigma_infty(
        const int axis_flag, const int mag,
        const int ispin, const std::vector<int>& wbands,
        DFT_output::atoms_info& atom,
        const std::vector<std::vector<Matrix>>& projector,
        Matrix& Sigma_infty)
  {
    debug::codestamp("self_energy::evalute_lattice_sigma_infty");

    const int nbands = wbands[ispin];
    const int iw = this->nomega(axis_flag)-1;
    const auto& local_sigma = this->correlated_sigma(axis_flag);
    Sigma_infty = Matrix::Zero(nbands, nbands);

    for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
    {
      const int iatom = atom.ineq_iatom(ineq);
      const int norb = atom.iatom_norb()[iatom];
      std::vector<int> equivalent_atoms{iatom};
      for(int a=0; a<atom.total_atoms(); a++)
        if(a != iatom && atom.equ_atom(a)==ineq) equivalent_atoms.push_back(a);

      for(int a : equivalent_atoms)
      {
        int spin = ispin;
        if(a != iatom){
          const int flip = mag==1 && atom.magnetic(iatom)*atom.magnetic(a)==-1;
          spin = (ispin+flip)%2;
        }
        const Matrix& P = projector[a][ispin];
        const Matrix& local = local_sigma[ineq][spin][iw];
        Eigen::VectorXcd diagonal = local.diagonal();
        // Legacy infty estimate discards Im Sigma only on representative atoms.
        if(a == iatom) diagonal = diagonal.real().cast<std::complex<double>>();
        const Matrix weighted_P = P * diagonal.asDiagonal();
        Sigma_infty += weighted_P * P.adjoint();
      }
    }
  }

  void self_energy::fit_sigma_static_from_tail(
      const std::vector<Matrix>& lattice_Sigma,
      const std::vector<double>& freq,
      int dim,
      int n_tail_fit,
      Matrix& out_sigma_stat)
    {
      const int nomega = (int)freq.size();
      if (nomega == 0) {
          out_sigma_stat = Matrix::Zero(dim, dim);
          return;
      }
      if ((int)lattice_Sigma.size() != nomega) {
          // size mismatch, fall back safely
          out_sigma_stat = Matrix::Zero(dim, dim);
          return;
      }
      for (int n = 0; n < nomega; n++) {
          if (lattice_Sigma[n].rows() != dim || lattice_Sigma[n].cols() != dim) {
              out_sigma_stat = Matrix::Zero(dim, dim);
              return;
          }
      }

      const int nfit = std::max(5, std::min(n_tail_fit, nomega));
      const int i0 = nomega - nfit;

      // x_t = 1/ω^2 (tail points)
      std::vector<double> x(nfit, 0.0);
      for (int t = 0; t < nfit; t++) {
          const double w = freq[i0 + t];
          // guard: avoid division by zero
          const double ww = (std::abs(w) < 1e-30) ? 1e-30 : w;
          x[t] = 1.0 / (ww * ww);
      }

      auto fit_intercept = [&](const std::vector<double>& y) -> double {
          // linear regression y = a + b x
          double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
          const int n = (int)y.size();
          for (int i = 0; i < n; i++) {
              sx  += x[i];
              sy  += y[i];
              sxx += x[i] * x[i];
              sxy += x[i] * y[i];
          }
          const double denom = n * sxx - sx * sx;
          if (std::abs(denom) < 1e-30) {
              return sy / n; // fallback: mean
          }
          const double b = (n * sxy - sx * sy) / denom;
          const double a = (sy - b * sx) / n;
          return a;
      };

      out_sigma_stat = Matrix::Zero(dim, dim);

      // Fit each matrix element independently using Re part
      std::vector<double> y(nfit, 0.0);
      for (int i = 0; i < dim; i++) {
          for (int j = 0; j < dim; j++) {
              for (int t = 0; t < nfit; t++) {
                  y[t] = lattice_Sigma[i0 + t](i,j).real();
              }
              const double a = fit_intercept(y);
              out_sigma_stat(i,j) = std::complex<double>(a, 0.0); // Im forced to 0
          }
      }

      // Hermitianize: (Σ + Σ†)/2
      for (int i = 0; i < dim; i++) {
          // diagonal strictly real
          out_sigma_stat(i,i) = std::complex<double>(out_sigma_stat(i,i).real(), 0.0);
          for (int j = i + 1; j < dim; j++) {
              const std::complex<double> sij = out_sigma_stat(i,j);
              const std::complex<double> sji = out_sigma_stat(j,i);
              const std::complex<double> sym = 0.5 * (sij + std::conj(sji));
              out_sigma_stat(i,j) = sym;
              out_sigma_stat(j,i) = std::conj(sym);
          }
      }
   }

  void self_energy::subtract_double_counting(const int axis_flag)
  {
    debug::codestamp("self_energy::subtract_double_counting");

    std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
            sigma_correlatd = this->correlated_sigma(axis_flag);
    
    const std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
            sigma_new = this->sigma_new(axis_flag);

    //Allocation
    if(sigma_correlatd.empty()) sigma_correlatd = sigma_new;
    // {
    //   sigma_correlatd.resize(this->dc.Vdc().size());
    //   for(int ineq=0; ineq<this->dc.Vdc().size(); ineq++)
    //   {
    //     sigma_correlatd[ineq].resize(this->dc.Vdc()[ineq].size());    
    //     for(int is=0; is<this->dc.Vdc().at(0).size(); is++)
    //     {
    //       sigma_correlatd[ineq][is].resize(this->nomega(axis_flag));
    //       for(int iomega=0; iomega<this->nomega(axis_flag); iomega++)
    //         sigma_correlatd[ineq][is][iomega].resize(this->dc.Vdc()[ineq][is].size());
    //     }//i_omega
    //   }//ineq
    // }

    for(int ineq=0; ineq<this->dc.Vdc().size(); ineq++)
    {
      const int nspin = this->dc.Vdc().at(0).size();
      const int omega_number = this->nomega(axis_flag);

      for(int ispin=0; ispin<nspin; ispin++)
      {
        const int m_index_number = this->dc.Vdc().at(ineq).at(ispin).size();
        for(int m_index=0; m_index<m_index_number; m_index++)
        {
          std::complex<double> tmp=this->dc.Vdc().at(ineq).at(ispin).at(m_index);
          for(int iomega=0; iomega<omega_number; iomega++){
            sigma_correlatd[ineq][ispin][iomega].data()[m_index] = sigma_new[ineq][ispin][iomega].data()[m_index] - tmp;
          }
        }//m_index
      }//ispin
    }//ineq

    return;  
  }

  int self_energy::nomega(const int axis_flag)
  {
    switch(axis_flag)
    {
    case 0:
      return this->sigma_imag.nomega();
      break;
    case 1:
      return this->sigma_real.nomega();
      break;
    default:
      std::cerr << "Error parameter of axis_flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
              self_energy::sigma_new(const int axis_flag)
  {
    switch(axis_flag)
    {
    case 0:
      return this->sigma_imag.sigma_new_access();
      break;
    case 1:
      return this->sigma_real.sigma_new_access();
      break;
    default:
      std::cerr << "Error parameter of axis_flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
            self_energy::sigma_save(const int axis_flag)
  {
    switch(axis_flag)
    {
    case 0:
      return this->sigma_imag.sigma_save_access();
      break;
    case 1:
      return this->sigma_real.sigma_save_access();
      break;
    default:
      std::cerr << "Error parameter of axis_flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<std::vector<std::vector<DFT_plus_DMFT::Matrix>>>&
              self_energy::correlated_sigma(const int axis_flag)
  {
    switch(axis_flag)
    {
    case 0:
      return this->sigma_imag.correlated_sigma_access();
      break;
    case 1:
      return this->sigma_real.correlated_sigma_access();
      break;
    default:
      std::cerr << "Error parameter of axis_flag" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  
}