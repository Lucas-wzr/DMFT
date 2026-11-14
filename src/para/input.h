#pragma once

#include <string>
#include <vector>

namespace DMFT
{
  struct input_parameters
  {
    // 第 1 类：条件必填参数，无默认值；无条件必填是条件恒为 true 的特例。
    double nf;                      //nominal occupation; required for double_counting=nominal, no default

    // 第 2 类：有默认值的参数，默认值在 read() 中设置。
    // Integer flags and hyb_func are internal representations of input options.
    int flag_DFT_solver;            //1:aims, 2:ABACUS
    double temperature;             //unit:Kelvin; default:300
    int flag_magnetism;             //AFM:1, FM:2, paramagenetism:3, none:4
    //ALPS-CTHYB:1;  LPS-CTHYB-SEGMENT:2; (not supported now)
    //PACS:3; Rutgers-CTHYB:4; iQIST:5; ComCTQMC:6; default:4
    int flag_impurity_solver;    
    int flag_interaction_type;        // density_density:1, kanamori:2, slater_condon:3, tensor:4; default:1
    int flag_double_counting;       //fll:1, nominal:2, arbitrary:3;   default:1
    int max_charge_step;            //number of the maximum charge self-consistent interation loop; default:1
    int max_dmft_step;              //number of the maximum DMFT interation step under each charge loop; default:1
    int max_dft_step;               //number of the maximum DFT interation step under each charge loop; default:1
    long long mc_step;              //number of MC sampling steps; default:5000000
    long long warmup;               //warmup updates per chain; currently used only by ComCTQMC; default:100000
    int comctqmc_nproc;             //0: use allocated physical cores on the launching node
    std::string comctqmc_exe, comctqmc_evalsim, comctqmc_launcher;
    std::vector<double> projection_window;  //projection_window: energy window for hybridization function
    bool hyb_func;                  //Whether hybrid functional is used or not; default:false
    double hyf_xc_alpha;            //The mixing factor in hybrid funtional; default:1.0;
    int flag_hyf_dc_mode;           //jacob:1; jacob_dp:2; default:1
    double hyf_dc_shift_d;          //extra shell-resolved DC shift for d shells in jacob_dp; input unit: eV
    double hyf_dc_shift_p;          //extra shell-resolved DC shift for p shells in jacob_dp; input unit: eV
    double charge_mix_param;         //The charge density mixing parameter; default:0.05
    double delta_sigma;             //The convergency criteria of self-energy (unit eV); default:0.1
    double delta_rho;               //The convergency criteria of the charge density; default: 1.0e-3
    int calculation_type;           //DFT+DMFT scf:0; spectra:1; default:0
    int mixing_step;                //Number of past iteration include in the density mixing; default:8

    // Four values supplied by the restart input keyword.
    int start_charge_step;          //The starting step of charge loop; default:1
    int start_DMFT_step;            //The stsrting step of DMFT loop; default:1
    int last_charge_step;           //The last one step before the starting step of charge loop; default:1
    int last_DMFT_step;             //The last one step before the stsrting step of DMFT loop; default:1

    int local_symmetry;             //0:none, 1:cubic, 2:t2g only, 3:eg only; default:0
    int flag_projector;             //nao (compatibility alias: default):1; cwf:2; default:nao
    double cwf_ep0;                 //lower CWF window edge relative to E_F; unit:eV; default:-5.0
    double cwf_ep1;                 //upper CWF window edge relative to E_F; unit:eV; default:5.0
    double cwf_t;                   //CWF window smoothing energy, independent of temperature; unit:eV; default:0.01
    double projection_min_weight;   //Minimum eigenvalue of the unweighted projector Gram matrix; default:0.3

    // 第 3 类：纯派生参数，不接受独立的 DMFT.in 输入。
    double beta;                   //from temperature: 1/(k_b*T); unit:1/Hartree
    // T in K, k_B in eV/K; truncate to int, then add 1 if odd.
    int n_tau;                     //CTQMC imaginary-time segments: int(25 eV/(k_B*T))
    int n_omega;                   //Matsubara frequency points: int(100 eV/(PI*k_B*T))
    bool flag_restart;             //true if start_charge_step>1 or start_DMFT_step>1

    // 第 4 类：可由用户覆盖的自动参数，未填写时根据其他参数补齐。
    int dmft_step_first;            //DMFT step limit under the first charge loop; default:max_dmft_step
    std::vector<double> dos_window;   //dos_window: energy window for DOS

    // 历史兼容成员：不参与上述有效参数分类，read() 忽略该输入。
    std::string dft_solver_exe;     //the executable of DFT solver

  };

  class input_info
  {
    public:
    bool read();

    static std::string strtolower(std::string in_str);

    const input_parameters& get() const {return params;}

    private:
    input_parameters params;
  };
}

// TODO: Choose strings or named enums for options as appropriate; avoid requiring
// all options to be numeric flags or introducing a full registration framework.
