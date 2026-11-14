#include "input.h"
#include "../constants.h"
#include "../debug.h"
#include "../mpi_environment.h"
#include "../global_variables.h"

#include <fstream>
#include <iostream>
#include <cstring>
#include <cctype>
#include <iomanip>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <string>
#include <map>
#include <set>
#include <sstream>
#include <limits>

namespace DMFT
{
  bool input_info::read()
  {
    debug::codestamp("input_info::read");

    // 步骤 1：仅为第 2 类参数设置固定默认值。
    // 第 1 类无默认值，第 4 类待依赖确定后补齐，第 3 类在校验后计算。
    params.calculation_type = 0;
    params.flag_DFT_solver = 1;
    params.temperature = 300.0;
    params.flag_magnetism = 3;
    params.flag_impurity_solver = 4;
    params.flag_interaction_type = 1;
    params.flag_double_counting = 1;
    params.projection_window = {-5.0, 5.0};
    params.max_charge_step = params.max_dmft_step = params.max_dft_step = 1;
    params.mc_step = 5000000;
    params.warmup = 100000;
    params.comctqmc_nproc = 0;
    params.comctqmc_exe = "CTQMC";
    params.comctqmc_evalsim = "EVALSIM";
    params.comctqmc_launcher = "comctqmc-launch";
    params.delta_sigma = 0.1;
    params.delta_rho = 1.0e-3;
    params.mixing_step = 8;
    params.charge_mix_param = 0.05;
    params.flag_projector = 1;
    params.cwf_ep0 = -5.0;
    params.cwf_ep1 = 5.0;
    params.cwf_t = 0.01;
    params.projection_min_weight = 0.3;
    params.hyb_func = false;
    params.hyf_xc_alpha = 1.0;
    params.flag_hyf_dc_mode = 1;
    params.hyf_dc_shift_d = params.hyf_dc_shift_p = 0.0;
    params.start_charge_step = params.start_DMFT_step = params.last_charge_step = params.last_DMFT_step = 1;
    params.local_symmetry = 0;

    // 步骤 2：一次读完文件，记录显式输入，再进行类型转换。
    std::map<std::string, std::vector<std::string>> input_entries;
    std::ifstream ifs("DMFT.in");
    if(!ifs) {
      std::cerr << "Fail to open file DMFT.in" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // fail：输出关键词 key 及错误原因 message，然后以失败状态退出程序。
    auto fail = [&](const std::string& key, const std::string& message) {
      std::cerr << "DMFT.in";
      std::cerr << ": " << key << ": " << message << std::endl;
      std::exit(EXIT_FAILURE);
    };

    std::string line;
    while(std::getline(ifs, line)) {
      const auto comment = line.find('#');
      if(comment != std::string::npos) line.erase(comment);
      std::istringstream tokens(line);
      std::string key, word;
      if(!(tokens >> key)) continue;
      key = strtolower(key);
      if(input_entries.count(key)) fail(key, "duplicate keyword");
      std::vector<std::string> words;
      while(tokens >> word) words.push_back(word);
      input_entries.emplace(key, words);
    }
    if(ifs.bad()) fail("input", "failed while reading file");

    // recognized 保存程序支持的关键词，后面用来检查未知关键词。
    std::set<std::string> recognized;
    // supplied：登记关键词 key，并判断用户是否提供了它。
    // 未提供返回 false；提供时检查值的个数是否为 count，正确才返回 true。
    auto supplied = [&](const std::string& key, int count) {
      recognized.insert(key);
      if(!input_entries.count(key)) return false;
      if(input_entries.at(key).size() != static_cast<std::size_t>(count))
        fail(key, "expected " + std::to_string(count) + " value(s)");
      return true;
    };

    // real：读取 key 对应的单个浮点数，写入 value 引用的原变量。
    auto real = [&](const std::string& key, double& value) {
      if(!supplied(key, 1)) return;
      const std::string& text = input_entries.at(key)[0];
      errno = 0;
      char* end = nullptr;
      const double parsed_input = std::strtod(text.c_str(), &end);
      if(end == text.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(parsed_input))
        fail(key, "invalid real value '" + text + "'");
      value = parsed_input;
    };
    // integer_value：将 text 严格转换为十进制 long long 整数并返回。
    // key 仅用于报错定位；此函数不查找输入项，也不设置默认值。
    auto integer_value = [&](const std::string& key, const std::string& text) {
      errno = 0;
      char* end = nullptr;
      const long long value = std::strtoll(text.c_str(), &end, 10);
      if(end == text.c_str() || *end != '\0' || errno == ERANGE)
        fail(key, "invalid integer value '" + text + "'");
      return value;
    };
    // integer：读取 key 对应的单个整数，检查 int 范围后写入 value。
    // 未提供时不修改 value；转换文本复用上面的 integer_value。
    auto integer = [&](const std::string& key, int& value) {
      if(!supplied(key, 1)) return;
      const long long parsed_input = integer_value(key, input_entries.at(key)[0]);
      if(parsed_input < std::numeric_limits<int>::min() || parsed_input > std::numeric_limits<int>::max())
        fail(key, "integer out of range");
      value = static_cast<int>(parsed_input);
    };
    // option：读取字符串选项并转成小写，在 options 的“名称—整数”表中匹配。
    // 匹配后将对应整数写入 value；未提供则不修改，不支持的名称报错。
    auto option = [&](const std::string& key, int& value,
                      std::initializer_list<std::pair<const char*, int>> options) {
      if(!supplied(key, 1)) return;
      const std::string parsed_input = strtolower(input_entries.at(key)[0]);
      for(const auto& choice : options)
        if(parsed_input == choice.first) { value = choice.second; return; }
      fail(key, "unsupported option '" + parsed_input + "'");
    };
    // window：读取两个浮点数，依次写入 value 的窗口下界和上界。
    // 未提供时不修改 value；这里只检查数值格式，窗口之间的约束在步骤 4 检查。
    auto window = [&](const std::string& key, std::vector<double>& value) {
      if(!supplied(key, 2)) return;
      value.resize(2);
      for(int i = 0; i < 2; ++i) {
        const std::string& text = input_entries.at(key)[i];
        errno = 0;
        char* end = nullptr;
        value[i] = std::strtod(text.c_str(), &end);
        if(end == text.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value[i]))
          fail(key, "invalid real value '" + text + "'");
      }
    };
    // text：读取 key 对应的单个字符串，原样保留，不做大小写规范化（可能是路径或可执行文件名）。
    auto text = [&](const std::string& key, std::string& value) {
      if(!supplied(key, 1)) return;
      value = input_entries.at(key)[0];
    };

    // 第 2 类：用户输入覆盖固定默认值；未填写时保留默认值。
    option("calculation", params.calculation_type, {{"scf", 0}, {"spectra", 1}});
    option("dft_solver", params.flag_DFT_solver, {{"aims", 1}, {"abacus", 2}});
    real("temperature", params.temperature);
    option("magnetism", params.flag_magnetism, {{"afm", 1}, {"fm", 2}, {"para", 3}, {"none", 4}});
    option("impurity_solver", params.flag_impurity_solver,
           {{"alps-cthyb", 1}, {"alps-cthyb-segment", 2}, {"pacs", 3}, {"rutgers-cthyb", 4}, {"iqist", 5},
            {"comctqmc", 6}});
    option("interaction_type", params.flag_interaction_type,
           {{"density_density", 1}, {"kanamori", 2}, {"slater_condon", 3}, {"tensor", 4}});
    option("double_counting", params.flag_double_counting, {{"fll", 1}, {"nominal", 2}, {"arbitrary", 3}});
    window("projection_window", params.projection_window);
    integer("max_charge_step", params.max_charge_step);
    integer("max_dmft_step", params.max_dmft_step);
    integer("max_dft_step", params.max_dft_step);
    real("delta_sigma", params.delta_sigma);
    real("delta_rho", params.delta_rho);
    integer("mixing_step", params.mixing_step);
    real("charge_mix_param", params.charge_mix_param);
    if(supplied("mc_step", 1)) params.mc_step = integer_value("mc_step", input_entries.at("mc_step")[0]);
    // ComCTQMC 的采样与启动参数：任何求解器下都接受这些关键词，取值只在 comctqmc 时生效。
    integer("comctqmc_nproc", params.comctqmc_nproc);
    if(supplied("warmup", 1)) params.warmup = integer_value("warmup", input_entries.at("warmup")[0]);
    text("comctqmc_exe", params.comctqmc_exe);
    text("comctqmc_evalsim", params.comctqmc_evalsim);
    text("comctqmc_launcher", params.comctqmc_launcher);
    integer("local_symmetry", params.local_symmetry);
    option("projector", params.flag_projector, {{"nao", 1}, {"default", 1}, {"cwf", 2}});
    real("cwf_ep0", params.cwf_ep0);
    real("cwf_ep1", params.cwf_ep1);
    real("cwf_t", params.cwf_t);
    real("projection_min_weight", params.projection_min_weight);
    // 保留现有 dft_xc 约定：只有 hybrid 启用杂化泛函。
    if(supplied("dft_xc", 1)) params.hyb_func = strtolower(input_entries.at("dft_xc")[0]) == "hybrid";
    real("hyf_xc_alpha", params.hyf_xc_alpha);
    option("hyf_dc_mode", params.flag_hyf_dc_mode,
           {{"jacob", 1}, {"jacob2008", 1}, {"jacob_dp", 2}, {"jacob-dp", 2}});
    real("hyf_dc_shift_d", params.hyf_dc_shift_d);
    real("hyf_dc_shift_p", params.hyf_dc_shift_p);
    if(supplied("restart", 4)) {
      int* steps[] = {&params.start_charge_step, &params.start_DMFT_step, &params.last_charge_step, &params.last_DMFT_step};
      for(int i = 0; i < 4; ++i) {
        const long long value = integer_value("restart", input_entries.at("restart")[i]);
        if(value < 1 || value > std::numeric_limits<int>::max()) fail("restart", "steps must be positive integers in range");
        *steps[i] = static_cast<int>(value);
      }
    }
    // 第 1 类：读取条件必填参数，不补默认值；必填条件在步骤 4 检查。
    real("nf", params.nf);

    // 第 4 类：先读取用户覆盖值；未填写的项在步骤 3 自动补齐。
    integer("dmft_step_first", params.dmft_step_first);
    window("dos_window", params.dos_window);

    // 第 3 类没有独立输入关键词，统一在步骤 5 计算。

    // 这两个历史关键词此前接受但不参与计算，暂保留兼容并明确提示。
    for(const auto& key : {"dft_solver_exe", "hybrid_xc_coeff"}) {
      recognized.insert(key);
      if(input_entries.count(key)) GLV::ofs_running << "Warning: " << key << " is ignored" << std::endl;
    }
    for(const auto& entry : input_entries)
      if(!recognized.count(entry.first)) fail(entry.first, "unsupported keyword");

    // 步骤 3：补齐第 4 类的自动默认值。只处理未显式输入的项，与文件行顺序无关。
    if(!input_entries.count("dos_window")) params.dos_window = params.projection_window;
    if(!input_entries.count("dmft_step_first")) params.dmft_step_first = params.max_dmft_step;

    // 步骤 4：检查第 1 类的必填条件，以及第 2、4 类的取值和组合约束。
    if(params.flag_double_counting == 2 && !input_entries.count("nf"))
      fail("nf", "required when double_counting is nominal; specify the nominal occupation explicitly");
    // 后端和求解器限制沿用原实现，不增加新的物理参数范围约束。
    #ifndef __FHIaims
      if(params.flag_DFT_solver == 1) fail("dft_solver", "FHI-aims support is not compiled in");
    #endif
    #ifndef __ABACUS
      if(params.flag_DFT_solver == 2) fail("dft_solver", "ABACUS support is not compiled in");
    #endif
    if(params.flag_impurity_solver == 1 || params.flag_impurity_solver == 2)
      fail("impurity_solver", "ALPS solvers are unsupported now");
    if(params.temperature <= 0.0) fail("temperature", "must be positive");
    if(params.dos_window[0] > params.projection_window[0] || params.dos_window[1] < params.projection_window[1])
      fail("dos_window", "must contain projection_window");
    if(params.flag_projector == 2 && (!(params.cwf_ep0 < params.cwf_ep1) || !(params.cwf_t > 0.0)))
      fail("projector", "CWF requires cwf_ep0 < cwf_ep1 and cwf_t > 0 (eV)");
    if(!(params.projection_min_weight > 0.0 && params.projection_min_weight <= 1.0))
      fail("projection_min_weight", "must be in (0,1]");
    if(params.flag_interaction_type != 1 && params.flag_impurity_solver != 4 && params.flag_impurity_solver != 6)
      fail("interaction_type", "general interactions require rutgers-cthyb or comctqmc");
    if(params.flag_impurity_solver == 6) {
      if(params.comctqmc_nproc < 0) fail("comctqmc_nproc", "must be non-negative");
      if(params.warmup < 0) fail("warmup", "must be non-negative");
      if(params.mc_step <= 0) fail("mc_step", "must be positive");
    }

    // 步骤 5：计算第 3 类纯派生量。温度用 K，beta 用 Hartree^-1。
    // 网格保持先截断、奇数再加一；转换前检查范围，避免整数溢出。
    params.beta = GLC::Hartree_to_eV/(params.temperature*GLC::K_BOLTZMAN_EV);
    const double tau_points = 25/(params.temperature*GLC::K_BOLTZMAN_EV);
    const double omega_points = 100/(GLC::PI*params.temperature*GLC::K_BOLTZMAN_EV);
    const int max_points = std::numeric_limits<int>::max() - 1;
    if(!std::isfinite(params.beta) || !std::isfinite(tau_points) || !std::isfinite(omega_points) ||
       tau_points > max_points || omega_points > max_points)
      fail("temperature", "derived imaginary-time/frequency grid exceeds integer range");
    params.n_tau = static_cast<int>(tau_points);
    params.n_omega = static_cast<int>(omega_points);
    if(params.n_tau % 2 == 1) ++params.n_tau;  // n_tau+1 为奇数，满足 Simpson 积分的网格约定。
    if(params.n_omega % 2 == 1) ++params.n_omega;
    params.flag_restart = params.start_charge_step > 1 || params.start_DMFT_step > 1;

    // 步骤 6：统一输出最终参数。保持原有关键词和值，原始 DMFT.in 不被覆盖。
    // 依赖 DFT 输出的量仍由后续初始化计算；运行步数仍由求解模块维护。
    GLV::ofs_running << "=========== All parameters used in the running time ===========\n"
      << std::setprecision(12)
      << "calculation  " << (params.calculation_type == 0 ? "scf" : "spectra") << '\n'
      << "dft_solver  " << (params.flag_DFT_solver == 1 ? "aims" : "abacus") << '\n'
      << "temperature  " << params.temperature << '\n'
      << "magnetism  " << std::vector<std::string>{"", "afm", "fm", "para", "none"}[params.flag_magnetism] << '\n'
      << "impurity_solver  " << std::vector<std::string>{"", "alps-cthyb", "alps-cthyb-segment", "pacs", "rutgers-cthyb", "iqist", "comctqmc"}[params.flag_impurity_solver] << '\n'
      << "interaction_type  " << std::vector<std::string>{"", "density_density", "kanamori", "slater_condon", "tensor"}[params.flag_interaction_type] << '\n'
      << "double_counting  " << std::vector<std::string>{"", "fll", "nominal", "arbitrary"}[params.flag_double_counting] << '\n';
    if(params.flag_double_counting == 2) GLV::ofs_running << "nf  " << params.nf << '\n';
    GLV::ofs_running
      << "projection_window  " << params.projection_window[0] << "  " << params.projection_window[1] << '\n'
      << "dos_window  " << params.dos_window[0] << "  " << params.dos_window[1] << '\n'
      << "max_charge_step  " << params.max_charge_step << '\n'
      << "max_dmft_step  " << params.max_dmft_step << '\n'
      << "dmft_step_first  " << params.dmft_step_first << '\n'
      << "max_dft_step  " << params.max_dft_step << '\n'
      << "delta_sigma  " << params.delta_sigma << '\n'
      << "delta_rho  " << params.delta_rho << '\n'
      << "mixing_step  " << params.mixing_step << '\n'
      << "charge_mix_param  " << params.charge_mix_param << '\n'
      << "mc_step  " << params.mc_step << '\n';
    if(params.flag_impurity_solver == 6) GLV::ofs_running
      << "comctqmc_nproc  " << params.comctqmc_nproc << '\n'
      << "warmup  " << params.warmup << '\n'
      << "comctqmc_exe  " << params.comctqmc_exe << '\n'
      << "comctqmc_evalsim  " << params.comctqmc_evalsim << '\n'
      << "comctqmc_launcher  " << params.comctqmc_launcher << '\n';
    GLV::ofs_running
      << "local_symmetry  " << params.local_symmetry << '\n'
      << "projector  " << (params.flag_projector == 1 ? "nao" : "cwf") << '\n'
      << "projection_min_weight  " << params.projection_min_weight << '\n';
    if(params.flag_projector == 2) GLV::ofs_running
      << "cwf_ep0  " << params.cwf_ep0 << " eV relative to E_F\n"
      << "cwf_ep1  " << params.cwf_ep1 << " eV relative to E_F\n"
      << "cwf_t  " << params.cwf_t << " eV\n";
    GLV::ofs_running
      << "hyf_xc_alpha  " << params.hyf_xc_alpha << '\n'
      << "hyf_dc_mode  " << (params.flag_hyf_dc_mode == 1 ? "jacob" : "jacob_dp") << '\n'
      << "hyf_dc_shift_d  " << params.hyf_dc_shift_d << " eV\n"
      << "hyf_dc_shift_p  " << params.hyf_dc_shift_p << " eV\n"
      << "restart  " << params.start_charge_step << " " << params.start_DMFT_step << " "
      << params.last_charge_step << " " << params.last_DMFT_step << '\n'
      << "===============================================================\n" << std::endl;
    return true;
  }

  std::string input_info::strtolower(std::string in_str)
  {
    // 按值传入，只修改字符串副本；tolower 要求字符先转换为 unsigned char。
    for(char& c : in_str)
      c = std::tolower(static_cast<unsigned char>(c));

    return in_str;
  }
}

// Parameter metadata and logging can later share a lightweight definition table.
