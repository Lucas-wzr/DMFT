# ABACUS 电荷接口脚本

这些脚本执行文件交换和单次 KS 更新；DMFT-dev C++ 主程序管理内外循环、密度矩阵及电荷混合。

- `prepare_abacus.py`：从参考 STRU 计算准确晶胞体积，准备初始 CSR 和 cube 的链接。
- `abacus_step.py`：准备一轮 ABACUS 输入，独立启动 ABACUS，转换输出并更新接口链接。
- `abacus_update.sh`：C++ 调用入口，启动同目录的 `abacus_step.py`。
- `convert_abacus.py`：转换 KS 能量、波函数、重叠矩阵和 k 点权重。
- `abacus_io.py`：共用的 INPUT、STRU 和 k 网格读取函数。

## 每个计算目录的设置

创建 `abacus_charge.json`，例如：

```json
{
  "reference": "abacus_reference",
  "environment_script": "/path/to/abacus_env.sh",
  "command": ["mpirun", "--mca", "ras", "^slurm", "--mca", "plm", "isolated", "--oversubscribe", "--bind-to", "none", "-np", "2", "abacus"]
}
```

`reference` 相对于计算目录解析，默认 `dft`。该目录应有参考 SCF 的 INPUT、STRU、KPT，以及 INPUT 指定的赝势和轨道目录；相对 `pseudo_dir`、`orbital_dir` 会转换成绝对路径。支持 INPUT 的 `stru_file` 和 `kpoint_file`，每轮复制后统一为 STRU 和 KPT。

`environment_script` 可省略，保留调用环境；`command` 是程序及参数的列表，默认 `["abacus"]`。上例是独立 GNU/OpenMPI 子进程的单节点启动方式，环境脚本路径及 MPI 进程数由计算目录决定，不再写死天津路径。也可通过 `environment` 对象提供环境变量字符串。

原来的 DMFT Intel MPI 进程启动状态会从子进程环境中清除。使用不同 MPI 实现时仍须提供 ABACUS 对应的环境脚本；多节点启动方式需另行验证。

## 初始密度和关联轨道

先完成参考 ABACUS SCF 并输出密度矩阵、cube、KS 波函数和 H/S。输出目录从 INPUT 的 `suffix` 推导，不限定材料名。

在计算目录执行：

```bash
python /path/to/DMFT-dev/scripts/prepare_abacus.py
```

脚本写入 `abacus_charge_volume.dat`（bohr³），并链接初始 `abacus_charge_reference.csr` 和 `abacus_charge_reference.cube`。已有参考文件保持原样。若自动识别不到输出，配置中可显式给出 `reference_density_matrix`、`reference_cube` 路径，两者也相对于计算目录解析。

关联原子、轨道、U/J 和等价关系由以下文件明确指定：

```text
dft/outputs_to_DMFT/correlated_atoms.info
dft/outputs_to_DMFT/symmetry.dat
```

转换器不会按元素猜测关联轨道，也不再生成 Ni 的固定索引。七个 f 轨道、多关联原子和不同 U/J 可由这些文件描述；实际 NAO 索引和实球谐顺序仍需对照该算例基组核实。

初次转换时，在计算目录的 `dft` 中放入实际 INPUT、STRU、KPT 和 `OUT.<suffix>`，先提供上述两份元数据，再执行：

```bash
python /path/to/DMFT-dev/scripts/convert_abacus.py
```

若这份接口需要作为不变的初始参考，将它放在单独目录，并让运行目录的 `dft/outputs_to_DMFT` 以符号链接指向它。每轮桥接会复制元数据到新接口，并只替换这个链接；不覆盖原始接口目录。

在计算目录创建通用入口的符号链接，供 C++ 调用：

```bash
ln -s /path/to/DMFT-dev/scripts/abacus_update.sh abacus_update.sh
```

入口默认使用 `DMFT-dev/python/bin/python`，该环境不存在时使用 `python3`；可通过 `DMFT_PYTHON` 指定解释器。脚本保持计算目录为当前工作目录。

每轮使用 `init_chg dm` 和 `scf_nmax 1`，输出至 `abacus_charge/stepN`。cube 统一复制成该轮目录下的 `charge.cube`，C++ 不再依赖材料的输出后缀。

## 数据约定与适用范围

- 当前支持固定结构、顺磁、非 SOC、`nspin=1`、`gamma_only=0`；没有新增 SOC 或磁性电荷后端。
- STRU 使用显式 `LATTICE_CONSTANT` 和 `LATTICE_VECTORS`。
- 支持任意尺寸的 Gamma、MP 网格以及显式 Direct/Cartesian 积分点。
- 从输入 KPT 恢复精确权重，从 WFC 头和 STRU 获取坐标，避免输出权重的小数截断。仅接受完整或时间反演约简网格；额外空间群约简会报错，因为还没有实现 NAO 矩阵的空间群旋转展开。建议 `symmetry 0` 或 `-1`。
- 显式 KPT 必须描述完整、时间反演对称的积分点及权重，不能只提供空间群不可约代表点或能带路径。
- 电子数优先读取 INPUT 的正值 `nelec`，否则读取 SCF 日志中的目标电子数。
- 同时支持旧式 `data-SR-sparse_SPIN0.csr`/`WFC_NAO_K*.txt` 和新式 `srs1_nao.csr`/`wfk*_nao.txt` 文件名。
- 原有正交性和 KS 残差检查保留，单位及傅里叶相位不变。

## 本地验证

```bash
python /path/to/DMFT-dev/scripts/test_abacus_interface.py
```

小文件测试覆盖 Gamma/MP 完整和时间反演约简网格、显式积分点、非对称网格拒绝、不同晶胞与后缀、七个 f 轨道元数据、参考文件准备和单步桥接。桥接测试使用模拟外部程序，不启动真实 ABACUS。

2026-10-09 使用 conda base 验证：NiO 的 666 k 点、40 基函数原始输出转换后，k 权重和能带数据与已有接口完全一致，k 坐标最大差 1.11e-16。报告见本地 `test_records/abacus_general_interface/nio`。未提交新的超算任务，未完成真实 Ce 的 DMFT 验证。
