> Warning: This demo **requires both calc-insight-kit AND logic-opt-kit**.
> 本案例同时依赖 calc-insight-kit（连续微积分可视化）与 logic-opt-kit（离散逻辑&约束优化求解），仅安装其中一个套件无法完整运行本示例。
>
> 定位：教学演示用途，非工业级求解器。源码遵循 MIT License。
>
> 学术诚信提示：本示例源码仅用于学习、课堂演示、原理探究参考。
> 禁止直接复制本脚本，不加修改直接作为课程实验报告、课程作业、课程论文的提交内容。
> 若用于你的作业/论文，请：重新推导问题、改写代码逻辑、调整模型参数、补充自己的分析与结论。
> 用户对自己提交的学术成果承担全部学术责任。

# 二维曲面约束优化可视化

## 教学知识点
1. calc-insight-kit：绘制二元目标函数三维连续曲面，直观观察无约束全局极值位置。
2. logic-opt-kit：定义一组线性不等式约束，求解可行域与约束条件下最优解。
3. 将求解得到的可行边界、约束最优点回传，叠加绘制到3D曲面对比展示。

## 学习观察要点
1. 无约束的极小值点是否落在约束可行域内部；
2. 当无约束最优点不可行时，约束最优解会落在约束边界上；
3. 直观理解：现实工程问题大量最优解被现实条件限制，不能取数学上纯粹的无约束极值。

## 运行依赖
```bash
export PYTHONPATH=.
pip install calc-insight-kit logic-opt-kit numpy matplotlib sympy highspy
```

运行 `demo.py`，自动输出 `output/` 目录。
