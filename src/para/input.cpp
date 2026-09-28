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

    // 步骤 1：设置固定默认值。依赖其他输入的默认值留到步骤 3。
    calculation_type = 0;
    flag_DFT_solver = 1;
    temperature = 300.0;
    flag_magnetism = 3;
    flag_impurity_solver = 4;
    flag_double_counting = 1;
    nf = 1.0;
    proj_window = {-5.0, 5.0};
    charge_step_max = DMFT_step_max = DFT_step_max = 1;
    MC_step = 5000000;
    delta_sigma = 0.1;
    delta_rho = 1.0e-3;
    mixing_step = 8;
    charge_mix_param = 0.05;
    flag_projector = 1;
    cwf_ep0 = -5.0;
    cwf_ep1 = 5.0;
    cwf_t = 0.01;
    hyb_func = false;
    hyf_xc_alpha = 1.0;
    flag_hyf_dc_mode = 1;
    hyf_dc_shift_d = hyf_dc_shift_p = 0.0;
    start_charge_step = start_DMFT_step = last_charge_step = last_DMFT_step = 1;
    int local_symmetry = 0;

    // 步骤 2：一次读完文件，记录显式输入，再进行类型转换。
    // 缺省不是异常；记录是否出现，避免用 0 或 -1 代替“未输入”。
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
    // 未提供时不修改 value；拒绝非法文本、超出范围的数值及 NaN/Inf。
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

    option("calculation", calculation_type, {{"scf", 0}, {"spectra", 1}});
    option("dft_solver", flag_DFT_solver, {{"aims", 1}, {"abacus", 2}});
    real("temperature", temperature);
    option("magnetism", flag_magnetism, {{"afm", 1}, {"fm", 2}, {"para", 3}, {"none", 4}});
    option("impurity_solver", flag_impurity_solver,
           {{"alps-cthyb", 1}, {"alps-cthyb-segment", 2}, {"pacs", 3}, {"rutgers-cthyb", 4}, {"iqist", 5}});
    option("double_counting", flag_double_counting, {{"fll", 1}, {"nominal", 2}, {"arbitrary", 3}});
    real("nf", nf);
    window("projection_window", proj_window);
    window("dos_window", DOS_window);
    integer("max_charge_step", charge_step_max);
    integer("max_dmft_step", DMFT_step_max);
    integer("dmft_step_first", DMFT_step_first);
    integer("max_dft_step", DFT_step_max);
    real("delta_sigma", delta_sigma);
    real("delta_rho", delta_rho);
    integer("mixing_step", mixing_step);
    real("charge_mix_param", charge_mix_param);
    if(supplied("mc_step", 1)) MC_step = integer_value("mc_step", input_entries.at("mc_step")[0]);
    integer("local_symmetry", local_symmetry);
    option("projector", flag_projector, {{"nao", 1}, {"default", 1}, {"cwf", 2}});
    real("cwf_ep0", cwf_ep0);
    real("cwf_ep1", cwf_ep1);
    real("cwf_t", cwf_t);
    // 保留现有 dft_xc 约定：只有 hybrid 启用杂化泛函。
    if(supplied("dft_xc", 1)) hyb_func = strtolower(input_entries.at("dft_xc")[0]) == "hybrid";
    real("hyf_xc_alpha", hyf_xc_alpha);
    option("hyf_dc_mode", flag_hyf_dc_mode,
           {{"jacob", 1}, {"jacob2008", 1}, {"jacob_dp", 2}, {"jacob-dp", 2}});
    real("hyf_dc_shift_d", hyf_dc_shift_d);
    real("hyf_dc_shift_p", hyf_dc_shift_p);
    if(supplied("restart", 4)) {
      int* steps[] = {&start_charge_step, &start_DMFT_step, &last_charge_step, &last_DMFT_step};
      for(int i = 0; i < 4; ++i) {
        const long long value = integer_value("restart", input_entries.at("restart")[i]);
        if(value < 1 || value > std::numeric_limits<int>::max()) fail("restart", "steps must be positive integers in range");
        *steps[i] = static_cast<int>(value);
      }
    }
    // 这两个历史关键词此前接受但不参与计算，暂保留兼容并明确提示。
    for(const auto& key : {"dft_solver_exe", "hybrid_xc_coeff"}) {
      recognized.insert(key);
      if(input_entries.count(key)) GLV::ofs_running << "Warning: " << key << " is ignored" << std::endl;
    }
    for(const auto& entry : input_entries)
      if(!recognized.count(entry.first)) fail(entry.first, "unsupported keyword");

    // 步骤 3：补齐依赖默认值。只处理未显式输入的项，与文件行顺序无关。
    if(!input_entries.count("dos_window")) DOS_window = proj_window;
    if(!input_entries.count("dmft_step_first")) DMFT_step_first = DMFT_step_max;

    // 步骤 4：在派生计算之前校验输入及组合。当前没有无条件必填项。
    // 后端和求解器限制沿用原实现，不增加新的物理参数范围约束。
    #ifndef __FHIaims
      if(flag_DFT_solver == 1) fail("dft_solver", "FHI-aims support is not compiled in");
    #endif
    #ifndef __ABACUS
      if(flag_DFT_solver == 2) fail("dft_solver", "ABACUS support is not compiled in");
    #endif
    if(flag_impurity_solver == 1 || flag_impurity_solver == 2)
      fail("impurity_solver", "ALPS solvers are unsupported now");
    if(temperature <= 0.0) fail("temperature", "must be positive");
    if(DOS_window[0] > proj_window[0] || DOS_window[1] < proj_window[1])
      fail("dos_window", "must contain projection_window");
    if(flag_projector == 2 && (!(cwf_ep0 < cwf_ep1) || !(cwf_t > 0.0)))
      fail("projector", "CWF requires cwf_ep0 < cwf_ep1 and cwf_t > 0 (eV)");

    // 步骤 5：计算纯派生量。温度用 K，beta 用 Hartree^-1。
    // 网格保持先截断、奇数再加一；转换前检查范围，避免整数溢出。
    beta = GLC::Hartree_to_eV/(temperature*GLC::K_BOLTZMAN_EV);
    const double tau_points = 25/(temperature*GLC::K_BOLTZMAN_EV);
    const double omega_points = 100/(GLC::PI*temperature*GLC::K_BOLTZMAN_EV);
    const int max_points = std::numeric_limits<int>::max() - 1;
    if(!std::isfinite(beta) || !std::isfinite(tau_points) || !std::isfinite(omega_points) ||
       tau_points > max_points || omega_points > max_points)
      fail("temperature", "derived imaginary-time/frequency grid exceeds integer range");
    n_tau = static_cast<int>(tau_points);
    n_omega = static_cast<int>(omega_points);
    if(n_tau % 2 == 1) ++n_tau;  // n_tau+1 为奇数，满足 Simpson 积分的网格约定。
    if(n_omega % 2 == 1) ++n_omega;
    flag_restart = start_charge_step > 1 || start_DMFT_step > 1;

    // 步骤 6：统一输出最终参数。保持原有关键词和值，原始 DMFT.in 不被覆盖。
    // 依赖 DFT 输出的量仍由后续初始化计算；运行步数仍由求解模块维护。
    GLV::ofs_running << "=========== All parameters used in the running time ===========\n"
      << std::setprecision(12)
      << "calculation  " << (calculation_type == 0 ? "scf" : "spectra") << '\n'
      << "dft_solver  " << (flag_DFT_solver == 1 ? "aims" : "abacus") << '\n'
      << "temperature  " << temperature << '\n'
      << "magnetism  " << std::vector<std::string>{"", "afm", "fm", "para", "none"}[flag_magnetism] << '\n'
      << "impurity_solver  " << std::vector<std::string>{"", "alps-cthyb", "alps-cthyb-segment", "pacs", "rutgers-cthyb", "iqist"}[flag_impurity_solver] << '\n'
      << "double_counting  " << std::vector<std::string>{"", "fll", "nominal", "arbitrary"}[flag_double_counting] << '\n';
    if(flag_double_counting == 2) GLV::ofs_running << "nf  " << nf << '\n';
    GLV::ofs_running
      << "projection_window  " << proj_window[0] << "  " << proj_window[1] << '\n'
      << "DOS_window  " << DOS_window[0] << "  " << DOS_window[1] << '\n'
      << "max_charge_step  " << charge_step_max << '\n'
      << "max_dmft_step  " << DMFT_step_max << '\n'
      << "dmft_step_first  " << DMFT_step_first << '\n'
      << "max_dft_step  " << DFT_step_max << '\n'
      << "delta_sigma  " << delta_sigma << '\n'
      << "delta_rho  " << delta_rho << '\n'
      << "mixing_step  " << mixing_step << '\n'
      << "charge_mix_param  " << charge_mix_param << '\n'
      << "mc_step  " << MC_step << '\n'
      << "local_symmetry  " << local_symmetry << '\n'
      << "projector  " << (flag_projector == 1 ? "nao" : "cwf") << '\n';
    if(flag_projector == 2) GLV::ofs_running
      << "cwf_ep0  " << cwf_ep0 << " eV relative to E_F\n"
      << "cwf_ep1  " << cwf_ep1 << " eV relative to E_F\n"
      << "cwf_t  " << cwf_t << " eV\n";
    GLV::ofs_running
      << "hyf_xc_alpha  " << hyf_xc_alpha << '\n'
      << "hyf_dc_mode  " << (flag_hyf_dc_mode == 1 ? "jacob" : "jacob_dp") << '\n'
      << "hyf_dc_shift_d  " << hyf_dc_shift_d << " eV\n"
      << "hyf_dc_shift_p  " << hyf_dc_shift_p << " eV\n"
      << "restart  " << start_charge_step << " " << start_DMFT_step << " "
      << last_charge_step << " " << last_DMFT_step << '\n'
      << "===============================================================\n" << std::endl;
    return true;
  }

  bool input_info::read_parameter(char* keyword, std::vector<std::string>& val, int count)
  {
    // 兼容 correlated_atoms 的独立读取；保留缺省时抛出 bool 的调用约定。
    // 与主读取流程一致，支持制表符、长行和末尾无换行。
    debug::codestamp("input_info::read_parameter");
    std::ifstream ifs("DMFT.in");
    if(!ifs) {
      std::cerr << "Fail to open file DMFT.in" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::set<std::string> known = {
      "calculation", "dft_solver", "dft_solver_exe", "temperature",
        "magnetism", "impurity_solver", "double_counting", "max_charge_step",
        "max_dmft_step", "dmft_step_first", "max_dft_step", "mc_step",
        "projection_window", "dos_window", "local_symmetry", "restart",
        "dft_xc", "charge_mix_param", "delta_sigma", "delta_rho",
        "mixing_step", "hybrid_xc_coeff", "hyf_xc_alpha", "hyf_dc_mode",
        "hyf_dc_shift_d", "hyf_dc_shift_p", "projector", "cwf_ep0",
        "cwf_ep1", "cwf_t", "nf"
    };
    std::string line;
    while(std::getline(ifs, line)) {
      const auto comment = line.find('#');
      if(comment != std::string::npos) line.erase(comment);
      std::istringstream tokens(line);
      std::string key, word;
      if(!(tokens >> key)) continue;
      key = strtolower(key);
      if(key == strtolower(keyword)) {
        std::vector<std::string> words;
        while(tokens >> word) words.push_back(word);
        if(words.size() != static_cast<std::size_t>(count)) throw line;
        val = words;
        return true;
      }
      if(!known.count(key)) throw std::string("Unsupported parameter " + key);
    }
    throw true;
  }

  std::string input_info::strtolower(std::string in_str)
  {
    // 按值传入，只修改字符串副本；tolower 要求字符先转换为 unsigned char。
    for(char& c : in_str)
      c = std::tolower(static_cast<unsigned char>(c));

    return in_str;
  }

  const void* input_info::parameter(char* keyword)
  {
    std::string word = this->strtolower(keyword);

    if(std::strcmp("dft_solver", word.c_str())==0) return &flag_DFT_solver;
    else if(std::strcmp("dft_solver_exe", word.c_str())==0) return &dft_solver_exe;
    else if(std::strcmp("calculation", word.c_str())==0) return &calculation_type;
    else if(std::strcmp("temperature", word.c_str())==0) return &temperature;
    else if(std::strcmp("beta", word.c_str())==0) return &beta;
    else if(std::strcmp("magnetism", word.c_str())==0) return &flag_magnetism;
    else if(std::strcmp("n_tau", word.c_str())==0) return &n_tau;
    else if(std::strcmp("n_omega", word.c_str())==0) return &n_omega;
    else if(std::strcmp("impurity_solver", word.c_str())==0) return &flag_impurity_solver;
    else if(std::strcmp("interaction_type", word.c_str())==0) return &flag_interaction_type;
    else if(word=="comctqmc_nproc") return &comctqmc_nproc;
    else if(word=="warmup") return &warmup;
    else if(word=="comctqmc_exe") return &comctqmc_exe;
    else if(word=="comctqmc_evalsim") return &comctqmc_evalsim;
    else if(word=="comctqmc_launcher") return &comctqmc_launcher;
    else if(std::strcmp("double_counting", word.c_str())==0) return &flag_double_counting;
    else if(std::strcmp("max_charge_step", word.c_str())==0) return &charge_step_max;
    else if(std::strcmp("max_dmft_step", word.c_str())==0) return &DMFT_step_max;
    else if(std::strcmp("max_dft_step", word.c_str())==0) return &DFT_step_max;
    else if(std::strcmp("mc_step", word.c_str())==0) return &MC_step;
    else if(std::strcmp("hyb_func", word.c_str())==0) return &hyb_func;
    else if(std::strcmp("hyf_xc_alpha", word.c_str())==0) return &hyf_xc_alpha;
    else if(std::strcmp("hyf_dc_shift_d", word.c_str())==0) return &hyf_dc_shift_d;
    else if(std::strcmp("hyf_dc_shift_p", word.c_str())==0) return &hyf_dc_shift_p;
    else if(std::strcmp("charge_mix_param", word.c_str())==0) return &charge_mix_param;
    else if(std::strcmp("delta_sigma", word.c_str())==0) return &delta_sigma;
    else if(std::strcmp("delta_rho", word.c_str())==0) return &delta_rho;
    else if(std::strcmp("start_charge_step", word.c_str())==0) return &start_charge_step;
    else if(std::strcmp("start_dmft_step", word.c_str())==0) return &start_DMFT_step;
    else if(std::strcmp("last_charge_step", word.c_str())==0) return &last_charge_step;
    else if(std::strcmp("last_dmft_step", word.c_str())==0) return &last_DMFT_step;
    else if(std::strcmp("restart", word.c_str())==0) return &flag_restart;
    else if(std::strcmp("mixing_step", word.c_str())==0) return &mixing_step;
    else if(std::strcmp("hyf_dc_mode", word.c_str())==0) return &flag_hyf_dc_mode;
    else if (std::strcmp("cwf_ep0", word.c_str()) == 0) return &cwf_ep0;
    else if (std::strcmp("cwf_ep1", word.c_str()) == 0) return &cwf_ep1;
    else if (std::strcmp("cwf_t", word.c_str()) == 0) return &cwf_t;
    else if (std::strcmp("projector", word.c_str()) == 0) return &flag_projector;
    else if (std::strcmp("projection_min_weight", word.c_str()) == 0) return &projection_min_weight;
    else if (std::strcmp("nf", word.c_str()) == 0) return &nf;
    else
    {
      std::cerr << "No parameter " << keyword << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }


}

// TODO: Replace the legacy const void* access interface with typed accessors.
// Parameter metadata and logging can later share a lightweight definition table.
// 将输出写得更完善
// 将内部的option对应的变量改用string
// 产出read_parameter
// 重构parameter方法，用struct保存和读取参数