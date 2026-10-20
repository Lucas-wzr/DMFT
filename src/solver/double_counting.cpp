#include "double_counting.h"
#include "../mpi_environment.h"
#include "../debug.h"
#include "../constants.h"
#include "../global_variables.h"
#include "Slater_Condon_parameterization.h"

#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace
{
  // For Slater-Condon interactions, U denotes F0, so the legacy conversion
  // from Kanamori U to averU does not apply. Use F0 and the shell-averaged J
  // in the DC formula; when explicit Slater integrals are supplied, derive
  // both values from them to keep DC consistent with the impurity interaction.
  // The returned vector contains {U_dc, J_dc} in Hartree.
  std::vector<double> slater_dc_parameters(int ineq, DFT_output::atoms_info& atom)
  {
    const int iatom = atom.ineq_iatom(ineq);
    const double U = atom.Uval(iatom), J = atom.Jval(iatom);
    // Slater U is F0; explicit integrals also determine the shell-averaged J.
    std::ifstream input("Slater_integrals_impurity" + std::to_string(ineq) + ".dat");
    if(!input) return {U,J};
    const int l = atom.L(iatom);
    const auto F = DMFT::slater_condon::read_integrals(input,l,GLC::Hartree_to_eV);
    double average_J;
    if(l == 1) average_J = F[1]/5.0;
    else if(l == 2) average_J = (F[1]+F[2])/14.0;
    else average_J = (286.0*F[1]+195.0*F[2]+250.0*F[3])/6435.0;
    return {F[0],average_J};
  }
}

namespace DMFT
{
  void double_counting::cal_double_counting(
      const int type, const bool SOC, const int nspin, 
      DFT_output::atoms_info& atom,
      DMFT::input_info& in,
      const bool hybrid_func,
      const double hyb_xc_coeff,
      const int hyf_dc_mode)
  {
    debug::codestamp("double_counting::cal_double_counting");

    const std::vector<int>& norb_sub = atom.iatom_norb();
    const std::complex<double> zero(0.0,0.0);

    std::vector<std::vector<std::vector<double>>> local_occ = atom.occ_num_m();

    if(type==1)
    {
      GLV::ofs_running << "\n=============Double counting(FLL)============" << std::endl;

      if(this->V_dc.empty()) this->V_dc.resize(atom.inequ_atoms());

      for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
      {
        const int iatom=atom.ineq_iatom(ineq);
        const int m_tot=norb_sub[iatom];

        int nspin_tmp=nspin;
        if(SOC) nspin_tmp=4;

        const double Norb = 2*m_tot;
        double averU = ( atom.Uval(iatom) + (atom.Uval(iatom)-2.0*atom.Jval(iatom))
                              *(Norb-2) )/(Norb-1.0);

        //K. Held. Adv. Phys., 56(2007):829--926
        // const double averU = ( atom.Uval(iatom) + (atom.Uval(iatom)-2.0*atom.Jval(iatom))*(m_tot-1.0) +
        //                       (atom.Uval(iatom)-3.0*atom.Jval(iatom))*(m_tot-1.0))/(2*m_tot-1.0);

        double averJ = atom.Jval(iatom);
        if(*(int*)in.parameter("interaction_type") == 3)
        {
          const auto parameters = slater_dc_parameters(ineq,atom);
          averU = parameters[0];
          averJ = parameters[1];
        }

        if(this->V_dc[ineq].empty()) this->V_dc[ineq].resize(nspin_tmp);
        
        for(int is=0; is<nspin_tmp; is++)
        {       
          if(this->V_dc[ineq][is].empty()) 
            this->V_dc[ineq][is].resize(m_tot*m_tot, zero);
 
          const double FLL_nominal = averU*(atom.occ_num(iatom,0)+atom.occ_num(iatom,1)-0.5)
                            -averJ*(atom.occ_num(iatom,is)-0.5);

          if(hybrid_func){
            if(hyf_dc_mode==1){
              // FLL_nominal + alpha * (U-J) * (0.5 - n_m_sigma).
              // When alpha=1, this becomes:
              //   U*(Nf-0.5) - J*(Nf_sigma-0.5) + (U-J)*(0.5-n_m_sigma)
              // = U*(Nf-n_m_sigma) - J*(Nf_sigma-n_m_sigma),
              // which is exactly the original Jacob 2008 form.
              for(int m=0; m<m_tot; m++){
                const double DC_HF = (averU-averJ) * (0.5 - local_occ[iatom][is][m]);

                this->V_dc.at(ineq).at(is).at(m*m_tot+m) =
                  std::complex<double>(FLL_nominal + hyb_xc_coeff*DC_HF, 0.0);
              }
            }
            else if(hyf_dc_mode==2){
              // Keep the Jacob-like alpha-scaled local term as the base part,
              // then add shell-resolved diagonal offsets for d and p shells to tune p-d alignment.
              const int angular_L = atom.L(iatom);
              const double shift_d_ev = *(double*)in.parameter("hyf_dc_shift_d");
              const double shift_p_ev = *(double*)in.parameter("hyf_dc_shift_p");
              double shell_shift_ev = 0.0;

              if(angular_L==2) shell_shift_ev = shift_d_ev;
              else if(angular_L==1) shell_shift_ev = shift_p_ev;

              const double shell_shift = shell_shift_ev / GLC::Hartree_to_eV;

              GLV::ofs_running << "  jacob_dp shell shift (impurity " << ineq
                               << ", spin " << is << ", l=" << angular_L << ") = "
                               << std::fixed << std::setprecision(6)
                               << shell_shift_ev << " eV" << std::endl;

              for(int m=0; m<m_tot; m++){
                const double DC_HF = (averU-averJ) * (0.5 - local_occ[iatom][is][m]);

                this->V_dc.at(ineq).at(is).at(m*m_tot+m) =
                  std::complex<double>(FLL_nominal + hyb_xc_coeff*DC_HF + shell_shift, 0.0);
              }
            }
            else{
              std::cerr << "Unsupported hyf_dc_mode" << std::endl;
              std::exit(EXIT_FAILURE);
            }
          }
          else{          
            for(int m=0; m<m_tot; m++)
              this->V_dc[ineq][is][m*m_tot+m] = std::complex<double>(FLL_nominal, 0.0);  //*atom.occ_num_m()[iatom][is][m];
          }
        
         GLV::ofs_running << "===========impurity:" << ineq << " spin:" << is << "========\n";
         
         for(int m=0; m<m_tot; m++)
           GLV::ofs_running << std::setw(12) << std::fixed << std::setprecision(6) << 
             GLC::Hartree_to_eV*this->V_dc[ineq][is][m*m_tot+m].real();

         GLV::ofs_running << std::endl;
         if(*(int*)in.parameter("interaction_type") == 3)
           GLV::ofs_running << "DC parameters: U_DC=" << averU*GLC::Hartree_to_eV
             << " eV J_DC=" << averJ*GLC::Hartree_to_eV << " eV N_DC="
             << atom.occ_num(iatom,0)+atom.occ_num(iatom,1)
             << " N_DC_spin=" << atom.occ_num(iatom,is) << '\n';
        
        }
      }//ineq

      /*
      for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
      {
        const int iatom=atom.ineq_iatom(ineq);
        const int angular_L=atom.L(iatom);
        const int m_tot = 2*angular_L+1;

        int nspin_tmp=nspin;
        if(SOC) nspin_tmp=4;

        std::vector<double> Nelec(2,0.0);
        for(int is=0; is<2; is++)
        {
          for(int m=0; m<m_tot; m++)
          {
            if(m==2 || m==4) continue;
            Nelec[is] += atom.occ_num_m()[iatom][is][m];
          }
        }

        this->V_dc.at(ineq).resize(nspin_tmp);
        for(int is=0; is<nspin_tmp; is++)
        {
          this->V_dc.at(ineq).at(is).resize(m_tot*m_tot,zero);
          const double tmp = atom.Uval(iatom)*(Nelec[0]+Nelec[1]-0.5)
                            -atom.Jval(iatom)*(Nelec[is]-0.5) ;
          for(int m=0; m<m_tot; m++)
          {
            if(m==2 || m==4) continue;
            this->V_dc.at(ineq).at(is).at(m*m_tot+m) = std::complex<double>(tmp,0.0)*atom.occ_num_m()[iatom][is][m];
          }
        }
      }//ineq
      */
    }//type==1
    else if(type==2)
    {
      GLV::ofs_running << "\n=============Double counting(nominal)============" << std::endl;

      if(this->V_dc.empty()) this->V_dc.resize(atom.inequ_atoms());

      const double nominal_nf = *(double*)in.parameter("nf");

      for(int ineq=0; ineq<atom.inequ_atoms(); ineq++)
      {
        const int iatom=atom.ineq_iatom(ineq);
        const int m_tot=norb_sub[iatom];

        int nspin_tmp=nspin;
        if(SOC) nspin_tmp=4;

        const double Norb = 2*m_tot;
        double averU = ( atom.Uval(iatom) + (atom.Uval(iatom)-2.0*atom.Jval(iatom))
                              *(Norb-2) )/(Norb-1.0);
        double averJ = atom.Jval(iatom);
        if(*(int*)in.parameter("interaction_type") == 3)
        {
          const auto parameters = slater_dc_parameters(ineq,atom);
          averU = parameters[0];
          averJ = parameters[1];
        }

        if(this->V_dc[ineq].empty()) this->V_dc[ineq].resize(nspin_tmp);

        // Nominal DC: use the input nominal occupation nf instead of the self-consistent one.
        // For hybrid_func && hyf_dc_mode==1, add the same Jacob-style orbital-dependent
        // HYF correction used in the FLL branch on top of the nominal average term.
        const double nominal_nf_sigma = nominal_nf / 2.0;
        const double Vdc_nominal =
          averU*(nominal_nf - 0.5) - averJ*(nominal_nf_sigma - 0.5);

        for(int is=0; is<nspin_tmp; is++)
        {
          if(this->V_dc[ineq][is].empty())
            this->V_dc[ineq][is].resize(m_tot*m_tot, zero);

          for(int m_index=0; m_index<m_tot*m_tot; m_index++)
            this->V_dc[ineq][is][m_index] = zero;

          if(hybrid_func && hyf_dc_mode==1){
            // Nominal + Jacob-style HYF correction:
            // keep the nominal average term from nf, then add the same
            // orbital-dependent Jacob contribution used in the HYF/FLL branch.
            // When hyf_xc_alpha = 1, this reduces to
            // U * (nf - n_m,sigma) - J * (nf_sigma - n_m,sigma).
            for(int m=0; m<m_tot; m++){
              const double DC_HF = (averU-averJ) * (0.5 - local_occ[iatom][is][m]);
              this->V_dc[ineq][is][m*m_tot+m] =
                std::complex<double>(Vdc_nominal + hyb_xc_coeff*DC_HF, 0.0);
            }
          }
          else{
            for(int m=0; m<m_tot; m++)
              this->V_dc[ineq][is][m*m_tot+m] = std::complex<double>(Vdc_nominal, 0.0);
          }

          GLV::ofs_running << "===========impurity:" << ineq << " spin:" << is << "========\n";
          for(int m=0; m<m_tot; m++)
            GLV::ofs_running << std::setw(12) << std::fixed << std::setprecision(6)
                             << GLC::Hartree_to_eV*this->V_dc[ineq][is][m*m_tot+m].real();
          GLV::ofs_running << std::endl;
          if(*(int*)in.parameter("interaction_type") == 3)
            GLV::ofs_running << "DC parameters: U_DC=" << averU*GLC::Hartree_to_eV
              << " eV J_DC=" << averJ*GLC::Hartree_to_eV << " eV N_DC="
              << nominal_nf << " N_DC_spin=" << nominal_nf_sigma << '\n';
        }
      }
    }
    else if(type==3)
    {
      std::cerr << "Arbitrary double counting is not implemented yet" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    else
    {
      std::cerr << "Unsupported double-counting type" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    return;
  }

}
