# ABACUS 电荷接口脚本

- `abacus_step.py`：主程序每轮调用一次，准备 ABACUS 输入、独立启动 KS 更新、调用转换脚本并更新计算目录的接口链接。
- `convert_abacus.py`：将 ABACUS 的 KS 能量、波函数和重叠矩阵转换为 DMFT-dev 的文件格式。

这两个脚本供主程序的 ABACUS 电荷后端使用，不管理外循环。当前仍使用天津服务器环境和 NiO 算例约定，尚未泛化到任意体系。

计算目录中的 `abacus_update.sh` 调用本目录的 `abacus_step.py`，并传入电荷步编号。`abacus_step.py` 将同目录的转换脚本复制到该轮输出目录后执行，以保留转换记录。
