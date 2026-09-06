> Warning: This demo **requires both calc-insight-kit AND logic-opt-kit**.
> 本案例同时依赖 calc-insight-kit（连续微积分可视化）与 logic-opt-kit（离散逻辑&约束优化求解），仅安装其中一个套件无法完整运行本示例。
>
> 定位：教学演示用途，非工业级求解器。源码遵循 MIT License。
>
> 学术诚信提示：本示例源码仅用于学习、课堂演示、原理探究参考。
> 禁止直接复制本脚本，不加修改直接作为课程实验报告、课程作业、课程论文的提交内容。
> 若用于你的作业/论文，请：重新推导问题、改写代码逻辑、调整模型参数、补充自己的分析与结论。
> 用户对自己提交的学术成果承担全部学术责任。

# 拉格朗日乘数法可视化演示

## 教学知识点
1. calc-insight-kit：绘制目标函数等值线（等高线）；绘制等式约束曲线。
2. logic-opt-kit：求解等式约束下的极值点（拉格朗日条件求解）。
3. 可视化展示核心几何结论：最优解处目标函数梯度与约束曲线法线共线。

## 学习观察要点
1. 观察等高线与约束曲线切点位置，即为等式约束极值；
2. 直观理解拉格朗日乘数的几何意义，不再只记忆代数公式；
3. 对比无约束极值点与等式约束下解的位置差异。

## 运行依赖
```bash
export PYTHONPATH=.
pip install calc-insight-kit logic-opt-kit numpy matplotlib sympy highspy
```

运行 `demo.py`，自动输出 `output/` 目录。
