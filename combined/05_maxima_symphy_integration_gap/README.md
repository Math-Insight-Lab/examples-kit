> Warning: This demo **requires both calc-insight-kit AND logic-opt-kit**.
> 本案例同时依赖 calc-insight-kit（连续微积分可视化）与 logic-opt-kit（离散逻辑&约束优化求解），仅安装其中一个套件无法完整运行本示例。
>
> 定位：教学演示用途，非工业级求解器。源码遵循 MIT License。
>
> 学术诚信提示：本示例源码仅用于学习、课堂演示、原理探究参考。
> 禁止直接复制本脚本，不加修改直接作为课程实验报告、课程作业、课程论文的提交内容。
> 若用于你的作业/论文，请：重新推导问题、改写代码逻辑、调整模型参数、补充自己的分析与结论。
> 用户对自己提交的学术成果承担全部学术责任。
>
># Maxima 与 SymPy 联合演示 — 双 CAS 互补求积分
>
>## 教学知识点
>1. **同一问题不同 CAS 引擎可能给出不同结果**：SymPy 基于 Risch 算法，Maxima 基于 Weierstrass 代换 + 经验规则，各有擅长与盲区。
>2. **SymPy 失败的案例，Maxima 可能成功求解**。本案例使用 `tan(x)/(cos(x)+1)` 作为测试算式：SymPy 返回未求值的 `Integral`，而 Maxima 返回精确的 `ln(cos(x)+1) - ln(cos(x))`。
>3. **数值交叉验证**：对 Maxima 的解析结果进行数值微分，与原始被积函数比较，误差 < 5.3e-11。
>4. **定积分数值验证**：`scipy.integrate.quad` 数值积分与解析结果差 8.33e-17。
>
>## 学习观察要点
>1. SymPy 返回 `Integral(...)` — 表示它**不能**在封闭形式内求出该积分；
>2. Maxima 返回 `ln(cos(x)+1) - ln(cos(x))` — 这是精确解析解；
>3. 对 Maxima 结果求导 `diff(result, x)`，误差 < 5.3e-11 — 验证了解析解的正确性；
>4. 观察两个引擎的求解时间差异。
>
>## 为什么需要双 CAS 引擎？
>| 特性 | SymPy | Maxima |
>|---|---|---|
>| 算法 | Risch 算法 + 启发式规则 | Weierstrass 代换 + 经验规则 |
>| 强项 | 符号化证明、代数操作、Python 生态集成 | 复杂积分/微分方程解析解 |
>| 本案例算式 | `Integral(tan(x)/(cos(x)+1), x)` 未求值 | 返回 `ln(cos(x)+1) - ln(cos(x))` |
>
>## 运行依赖
>```bash
>export PYTHONPATH=.
>pip install calc-insight-kit logic-opt-kit numpy matplotlib sympy highspy scipy
>```
>
>运行 `demo.py`，自动输出 `output/` 目录。
