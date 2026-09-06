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
># Maxima 原函数 × 逻辑优化器 — 跨库协同最优化
>
>## 教学知识点
>本案例演示 **calc-insight-kit（连续微积分）与 logic-opt-kit（离散逻辑&约束优化）的真正跨库协同**：
>1. 用 Maxima（通过 calc-insight-kit）求解一个连续函数的原函数（不定积分），作为"收益/代价函数"。
>2. 将积分结果离散化为 10 个区间的"收益值"。
>3. 用 logic-opt-kit 的逻辑约束 + MIP 求解器，在给定总预算限制下，选出总收益最大的若干区间。
>
>## 数学建模流程
>1. **连续分析**：对 `f(x) = 0.5*x^3 - 3*x^2 + 4*x + 1` 求原函数 `F(x)`；
>2. **离散化**：计算区间 `[0,1], [1,2], ..., [9,10]` 上的收益 `F(i+1)-F(i)`；
>3. **逻辑约束建模**：`x0 + x1 + ... + x9 <= 5`（最多选5个区间），`x_i in {0,1}`；
>4. **MIP 求解**：用 logic-opt-kit 求解最大化总收益的区间选择。
>
>## 学习观察要点
>1. 观察哪些区间的收益最高，MIP 求解器如何"聪明地"选择它们；
>2. 理解连续微积分（原函数）如何与离散优化（0-1 选择）无缝衔接；
>3. 体会"数学分析 → 离散建模 → 数值求解"的完整建模链路。
>
>## 跨库协作架构图
>```
>f(x) = 0.5x³ - 3x² + 4x + 1
>    │
>    ▼
>calc-insight-kit (Maxima CAS)
>    ├── integrate(f(x), x) → F(x)
>    └── 离散化 → gains[i] = F(i+1) - F(i)
>    │
>    ▼
>logic-opt-kit (MIP Solver)
>    ├── QuickModelBuilder
>    ├── 0-1 变量 + 预算约束
>    └── solve() → 最优区间选择
>    │
>    ▼
>可视化：原函数曲线 + 选中区间高亮
>```
>
>## 运行依赖
>```bash
>export PYTHONPATH=.
>pip install calc-insight-kit logic-opt-kit numpy matplotlib sympy highspy
>```
>
>运行 `demo.py`，自动输出 `output/` 目录。
