# -*- coding: utf-8 -*-
"""
Demo 3 — Production-Budget Optimization
=========================================
Cross-library demo: calc-insight-kit (2D visualization, SymPy solver)
                     + logic-opt-kit (LP solver for linear constraints)

Problem (Math Modeling Classic):
  A factory produces two products A and B.
    Product A: profit = 50x - 0.5x^2  (diminishing returns)
    Product B: profit = 40y - 0.3y^2
  Total profit: P(x,y) = 50x - 0.5x^2 + 40y - 0.3y^2

  Constraints:
    2x + y <= 100   (raw material, kg)
    x + 2y <= 80    (labor hours)
    x + y <= 60     (total capacity, units)
    x >= 0, y >= 0

  Find: (x, y) that maximizes total profit.
"""
import os
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calc_insight_kit import use_theme
from logic_opt_kit.core.solver_backend import get_solver_backend
from logic_opt_kit.core.model_builder import QuickModelBuilder
from logic_opt_kit.viz import plot_solution_bar

use_theme("teaching")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Demo 3: Production-Budget Optimization")
    print("=" * 60)

    # ── Step 1: Problem definition ──
    print("\n[1] Problem Definition:")
    print("    Maximize:  P(x,y) = 50x - 0.5x^2 + 40y - 0.3y^2")
    print("    Subject to:")
    print("      2x + y <= 100   (raw material)")
    print("      x + 2y <= 80    (labor hours)")
    print("      x + y <= 60     (capacity)")
    print("      x >= 0, y >= 0")

    # ── Step 2: Analytical solution with SymPy ──
    print("\n[2] Analytical solution with SymPy:")
    x, y = sp.symbols('x y', real=True)
    P = 50*x - 0.5*x**2 + 40*y - 0.3*y**2

    # Unconstrained maximum
    dPdx = sp.diff(P, x)
    dPdy = sp.diff(P, y)
    unconstrained = sp.solve([dPdx, dPdy], (x, y))
    print(f"    Unconstrained max: x={float(unconstrained[x]):.2f}, y={float(unconstrained[y]):.2f}")
    print(f"    Profit at unconstrained: {float(P.subs(unconstrained)):.2f}")

    # ── Step 3: Linear approximation for LP solver ──
    # Since logic-opt-kit uses LP (linear programming), we linearize the
    # quadratic objective at a reasonable guess point and iterate.
    print("\n[3] Solving with logic-opt-kit (LP with gradient linearization):")

    # First iteration: linearize at origin
    # grad P at (0,0): dP/dx = 50, dP/dy = 40
    backend = get_solver_backend("highs")
    mb = QuickModelBuilder(backend, model_name="production")

    x1 = mb.var("x", low=0, high=None, vtype="C")
    y1 = mb.var("y", low=0, high=None, vtype="C")

    mb.cons(2*x1 + y1 <= 100, name="material")
    mb.cons(x1 + 2*y1 <= 80, name="labor")
    mb.cons(x1 + y1 <= 60, name="capacity")
    mb.obj(50*x1 + 40*y1, sense="max")  # linear approx at origin
    mb.solve()
    sol1 = mb.solution()
    print(f"    Iteration 1 (linear at origin): x={sol1['x']:.2f}, y={sol1['y']:.2f}")

    # Second iteration: linearize at solution of iteration 1
    # dP/dx = 50 - x, dP/dy = 40 - 0.6*y
    grad_x = 50 - sol1['x']
    grad_y = 40 - 0.6 * sol1['y']
    print(f"    Gradient update: dP/dx={grad_x:.2f}, dP/dy={grad_y:.2f}")

    backend2 = get_solver_backend("highs")
    mb2 = QuickModelBuilder(backend2, model_name="production_iter2")
    x2 = mb2.var("x", low=0, high=None, vtype="C")
    y2 = mb2.var("y", low=0, high=None, vtype="C")
    mb2.cons(2*x2 + y2 <= 100, name="material")
    mb2.cons(x2 + 2*y2 <= 80, name="labor")
    mb2.cons(x2 + y2 <= 60, name="capacity")
    mb2.obj(grad_x * x2 + grad_y * y2, sense="max")
    mb2.solve()
    sol2 = mb2.solution()
    print(f"    Iteration 2: x={sol2['x']:.2f}, y={sol2['y']:.2f}")

    # Compute actual quadratic profit at the LP solution
    actual_profit = (50*sol2['x'] - 0.5*sol2['x']**2
                     + 40*sol2['y'] - 0.3*sol2['y']**2)
    print(f"    Actual profit at LP solution: {actual_profit:.2f}")

    # Compare with exact analytical solution using SymPy KKT conditions
    print("\n[4] Exact solution via SymPy KKT conditions:")
    lam1, lam2, lam3 = sp.symbols('lam1 lam2 lam3', nonnegative=True)

    # KKT: gradient P = lam1*grad(g1) + lam2*grad(g2) + lam3*grad(g3)
    # g1 = 100 - 2x - y >= 0, g2 = 80 - x - 2y >= 0, g3 = 60 - x - y >= 0
    KKT_eqs = [
        dPdx - 2*lam1 - lam2 - lam3,   # dP/dx = 2*lam1 + lam2 + lam3
        dPdy - lam1 - 2*lam2 - lam3,    # dP/dy = lam1 + 2*lam2 + lam3
    ]
    print(f"    KKT stationarity: {KKT_eqs}")
    print(f"    (Solving KKT system requires case analysis — using numerical evaluation)")

    # Numerical verification: evaluate P on constraint vertices
    # Vertex 1: 2x+y=100, x+2y=80 => x=40, y=20 => x+y=60 ✓
    v1 = (40, 20)
    # Vertex 2: 2x+y=100, x+y=60 => x=40, y=20 (same as above)
    # Vertex 3: x+2y=80, x+y=60 => y=20, x=40 (same)
    # Vertex 4: x=0, x+y=60 => x=0, y=60 => 2*0+60<=100, 0+120>80 ✗
    # Vertex 5: x=0, x+2y=80 => x=0, y=40 => 0+40<=60 ✓
    v5 = (0, 40)
    # Vertex 6: x+2y=80, y=0 => x=80, y=0 => 160>100 ✗
    # Vertex 7: 2x+y=100, y=0 => x=50, y=0 => 50<=60 ✓
    v7 = (50, 0)
    # Vertex 8: x=0, y=0
    v8 = (0, 0)

    vertices = [v5, v7, v8]
    best_v = None
    best_p = -float('inf')
    for vx, vy in vertices:
        pv = 50*vx - 0.5*vx**2 + 40*vy - 0.3*vy**2
        print(f"    Vertex ({vx}, {vy}): P = {pv:.2f}")
        if pv > best_p:
            best_p = pv
            best_v = (vx, vy)
    print(f"    Best vertex: {best_v}, P = {best_p:.2f}")

    # ── Step 4: Visualization ──
    print("\n[5] Generating visualization...")

    fig = plt.figure(figsize=(14, 6))

    # --- 2D profit contour ---
    ax1 = fig.add_subplot(121)
    x_vals = np.linspace(0, 60, 100)
    y_vals = np.linspace(0, 50, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = 50*X - 0.5*X**2 + 40*Y - 0.3*Y**2

    contour = ax1.contour(X, Y, Z, levels=15, cmap='YlOrRd')
    ax1.clabel(contour, inline=True, fontsize=7, fmt='%.0f')

    # Constraints
    ax1.plot([0, 50], [100, 0], 'b-', linewidth=2, label='2x+y<=100')
    ax1.plot([80, 0], [0, 40], 'g-', linewidth=2, label='x+2y<=80')
    ax1.plot([60, 0], [0, 60], 'orange', linewidth=2, label='x+y<=60')

    # Feasible region (approximate)
    ax1.fill_between([0, 30, 40, 0], [0, 40, 20, 0], alpha=0.15,
                      color='purple', label='Feasible Region')

    # Mark solutions
    ax1.plot(float(unconstrained[x]), float(unconstrained[y]), 'ko',
             markersize=10, label='Unconstrained Max')
    ax1.plot(sol2['x'], sol2['y'], 'r*', markersize=15, label='LP Solution')
    if best_v:
        ax1.plot(best_v[0], best_v[1], 'ms', markersize=10,
                label=f'Best Vertex {best_v}')

    ax1.set_xlabel('Product A (x)')
    ax1.set_ylabel('Product B (y)')
    ax1.set_title('Profit Surface Contour + Constraints')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Sensitivity analysis ---
    ax2 = fig.add_subplot(122)
    budgets = np.linspace(50, 150, 30)
    profits = []
    for budget in budgets:
        backend_s = get_solver_backend("highs")
        mb_s = QuickModelBuilder(backend_s, model_name=f"sensitivity_{budget:.0f}")
        xs = mb_s.var("x", low=0, high=None, vtype="C")
        ys = mb_s.var("y", low=0, high=None, vtype="C")
        mb_s.cons(2*xs + ys <= budget, name="mat")
        mb_s.cons(xs + 2*ys <= 80, name="lab")
        mb_s.cons(xs + ys <= 60, name="cap")
        mb_s.obj(50*xs + 40*ys, sense="max")
        mb_s.solve()
        ss = mb_s.solution()
        profits.append(50*ss['x'] - 0.5*ss['x']**2 + 40*ss['y'] - 0.3*ss['y']**2)

    ax2.plot(budgets, profits, 'b-', linewidth=2)
    ax2.set_xlabel('Raw Material Budget (kg)')
    ax2.set_ylabel('Total Profit')
    ax2.set_title('Sensitivity: Profit vs. Raw Material Budget')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "production_budget_demo.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")

    # --- Solution bar chart ---
    plot_solution_bar(["Product A", "Product B"],
                      [sol2['x'], sol2['y']],
                      save_path=os.path.join(OUTPUT_DIR, "production_solution.png"))
    print(f"    Saved: {os.path.join(OUTPUT_DIR, 'production_solution.png')}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  • calc-insight-kit (SymPy) found unconstrained max at x={float(unconstrained[x]):.1f}, y={float(unconstrained[y]):.1f}")
    print(f"  • logic-opt-kit (HiGHS LP) approximated quadratic with iterative linearization")
    print(f"  • LP solution: x={sol2['x']:.1f}, y={sol2['y']:.1f}")
    print(f"  • Best vertex found: {best_v}, P={best_p:.2f}")
    print("  • This is a classic operations research / management science problem")
    print("=" * 60)


if __name__ == "__main__":
    main()
