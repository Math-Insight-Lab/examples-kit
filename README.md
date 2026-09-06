# examples-kit
Cross-library combined demo repository for calc-insight-kit & logic-opt-kit

### AI‑Assisted Code Notice
> 中文：本项目部分代码片段由AI大模型辅助生成，所有AI产出均经过人工审阅、改写、调试验证。项目全部代码统一遵循 MIT License。AI仅作为编码辅助工具，整体架构与核心设计由人类开发者完成。使用本项目请自行承担技术风险。

> English: Portions of source code are AI‑assisted generated. All AI outputs have been manually reviewed, rewritten and tested. The entire project is licensed under MIT License. AI serves only as coding assistant; overall architecture and core design are created by human maintainer. Use at your own risk.

LICENSE MIT

> Warning: This repository **ONLY contains combined demos that require BOTH calc-insight-kit AND logic-opt-kit**.
> Stand-alone examples for each library are kept inside each main repository, source-code is NOT copied to this repo.

## 1. Combined cross-library demos (need two libraries installed together)

Folder: [combined](./combined)

List of joint teaching demos:

- 01_constrained_2d_optimization: Constrained 2D optimization — objective function surface with constraint boundary overlay
- 02_lagrange_multiplier_demo: Lagrange multiplier visualization — contour lines and equality constraint tangency
- 03_mathmodel_production_budget: Production-budget math modeling — multi-constraint economic optimization
- 04_logic_condition_domain_filter: Logic-condition domain filter — plot only where logical predicates hold

> Local run hint:
> You **MUST set PYTHONPATH=. at repository root** before executing demos.
>
> ```bash
> # Linux / macOS
> export PYTHONPATH=.
> python combined/01_constrained_2d_optimization/demo.py
>
> # Windows PowerShell
> $env:PYTHONPATH="."
> python combined/01_constrained_2d_optimization/demo.py
> ```

Install dependencies for combined demos:

```bash
pip install -r requirements-combined.txt
```

> Note: Each core library can be installed and used independently. Combined demos require both libraries.

## 2. Stand-alone single-library examples (Jump link, no source code here)

### calc-insight-kit standalone examples

Link: calc-insight-kit/tree/main/examples

> kid / highschool / university demos, only need calc-insight-kit.

### logic-opt-kit standalone examples

Link: logic-opt-kit/tree/main/examples

> logic & pure-optimization demos, only need logic-opt-kit.

---

## Academic Notice

All example scripts in this repository are teaching-demonstration samples.
本仓库全部示例脚本为教学演示样例。

1. Allowed usage: personal learning, classroom demonstration, course reference, inspiration for modeling ideas.

2. Prohibited: Directly copy source code without modification and submit as course experiment, homework or paper.
禁止直接复制源码、不作修改直接提交课程实验、作业、课程论文。

3. If you reference these scripts in academic work:
    - Modify model or parameters;
    - Rewrite partial implementation logic;
    - Add your own derivation, observation analysis and conclusion.

4. Users take full academic responsibility for their submitted works.
使用者对自己提交的学术成果承担全部学术责任。
