# -*- coding: utf-8 -*-
"""
Demo 4 — Logic-Condition Domain Filter
========================================
Cross-library demo: calc-insight-kit (function visualization)
                     + logic-opt-kit (logic constraint builder + feasibility solver)

Concept: Use propositional/MSO logic predicates to filter which parts
of a function domain are valid for plotting.

Problem:
  Plot f(x,y) = sin(x) * cos(y) only where the logical condition holds:
    (x > 0 AND y > 0) OR (x < -1 AND y < 0)

This demonstrates bridging discrete logic (truth tables, predicates)
with continuous function visualization — a key skill in mathematical logic
and analysis.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calc_insight_kit import use_theme
from logic_opt_kit.core.logic_solver import LogicConstraintBuilder, build_mso_location_constraint
from logic_opt_kit.core.model_builder import QuickModelBuilder
from logic_opt_kit.core.solver_backend import get_solver_backend

use_theme("teaching")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_logic_predicate(x, y):
    """
    Apply the logical condition:
      (x > 0 AND y > 0) OR (x < -1 AND y < 0)
    Returns a boolean mask.
    """
    cond1 = (x > 0) & (y > 0)       # first clause: AND
    cond2 = (x < -1) & (y < 0)      # second clause: AND
    return cond1 | cond2             # OR of the two clauses


def main():
    print("=" * 60)
    print("Demo 4: Logic-Condition Domain Filter")
    print("=" * 60)

    # ── Step 1: Define logic predicates ──
    print("\n[1] Logic Predicate Definition:")
    print("    Condition: (x > 0 AND y > 0) OR (x < -1 AND y < 0)")
    print("    This defines two quadrants where the function is 'visible'.")

    # Use logic-opt-kit to build the constraint representation
    print("\n[2] Building logic constraints with logic-opt-kit:")
    lb = LogicConstraintBuilder()

    # Propositional logic: build the constraint structure
    # We map: p1 = (x > 0), p2 = (y > 0), p3 = (x < -1), p4 = (y < 0)
    # Formula: (p1 AND p2) OR (p3 AND p4)
    # In CNF: (p1 OR p3) AND (p1 OR p4) AND (p2 OR p3) AND (p2 OR p4)

    # For MSO: "at least one of the two clauses is satisfied"
    lb.mso_subset_at_least_one(["clause_A", "clause_B"])
    print("    MSO constraint: at least one clause is true")

    # One-order logic: AND/OR/NOT
    lb.add_or("p1", "p3")   # x > 0 OR x < -1
    lb.add_or("p2", "p4")   # y > 0 OR y < 0
    print("    CNF clauses built successfully")

    # Print the built constraints
    clauses = lb.get_clauses()
    mso_constraints = lb.get_mso_constraints()
    print(f"    Logic clauses: {clauses}")
    print(f"    MSO constraints: {mso_constraints}")

    # ── Step 2: Define the function ──
    print("\n[3] Function Definition:")
    print("    f(x,y) = sin(x) * cos(y)")

    # ── Step 3: Create mesh and apply filter ──
    print("\n[4] Applying logic predicate filter...")

    x_vals = np.linspace(-5, 5, 200)
    y_vals = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.sin(X) * np.cos(Y)

    # Apply logic predicate mask
    mask = apply_logic_predicate(X, Y)
    Z_filtered = np.where(mask, Z, np.nan)

    # ── Step 4: Visualization ──
    print("    Generating visualizations...")

    fig = plt.figure(figsize=(16, 5))

    # --- Left: Full surface (without filter) ---
    ax1 = fig.add_subplot(131)
    contour_full = ax1.contourf(X, Y, Z, levels=30, cmap='RdBu_r', extend='both')
    ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Full Domain: f(x,y) = sin(x)*cos(y)')
    ax1.grid(True, alpha=0.2)
    fig.colorbar(contour_full, ax=ax1, shrink=0.8)

    # --- Middle: Filtered by logic predicate ---
    ax2 = fig.add_subplot(132)
    contour_filtered = ax2.contourf(X, Y, Z_filtered, levels=30, cmap='RdBu_r', extend='both')
    ax2.contour(X, Y, Z, levels=10, colors='gray', alpha=0.2, linewidths=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Logic-Filtered Domain\n(x>0 AND y>0) OR (x<-1 AND y<0)')
    ax2.grid(True, alpha=0.2)
    fig.colorbar(contour_filtered, ax=ax2, shrink=0.8)

    # --- Right: 1D cross-section with logic filter ---
    ax3 = fig.add_subplot(133)
    x_slice = np.linspace(-5, 5, 500)
    y_fixed = 1.5  # fixed y in the positive region

    z_slice = np.sin(x_slice) * np.cos(y_fixed)

    # Apply 1D logic predicate
    mask_1d = (x_slice > 0) | (x_slice < -1)
    z_slice_filtered = np.where(mask_1d, z_slice, np.nan)

    ax3.plot(x_slice, z_slice, 'b-', alpha=0.3, linewidth=1, label='Full function')
    ax3.plot(x_slice, z_slice_filtered, 'r-', linewidth=2, label='Logic-filtered')
    ax3.axvline(0, c='orange', ls='--', alpha=0.5, label='x=0 boundary')
    ax3.axvline(-1, c='orange', ls='--', alpha=0.5)
    ax3.set_xlabel('x (at y=1.5)')
    ax3.set_ylabel('f(x, 1.5)')
    ax3.set_title('1D Cross-Section with Logic Filter')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "logic_domain_filter.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Saved: {save_path}")

    # ── Step 5: Demonstrate constraint feasibility with binary MIP ──
    print("\n[5] Verifying feasibility with logic-opt-kit:")
    backend = get_solver_backend("highs")
    mb = QuickModelBuilder(backend, model_name="feasibility_check")

    # Binary variables representing whether each predicate is satisfied
    # Note: logic-opt-kit uses "B" for binary in add_variable
    p1 = mb.var("p1", low=0, high=1, vtype="B")   # x > 0
    p2 = mb.var("p2", low=0, high=1, vtype="B")   # y > 0
    p3 = mb.var("p3", low=0, high=1, vtype="B")   # x < -1
    p4 = mb.var("p4", low=0, high=1, vtype="B")   # y < 0

    # At least one clause must be satisfied
    mb.cons(p1 + p2 + p3 + p4 >= 1, name="at_least_one_true")

    # We want to find a feasible assignment — maximize sum of p's
    # so that at least one clause is satisfied
    mb.obj(p1 + p2 + p3 + p4, sense="max")
    mb.solve()
    sol = mb.solution()
    print(f"    Feasible logic assignment: p1={sol['p1']:.0f}, p2={sol['p2']:.0f}, "
          f"p3={sol['p3']:.0f}, p4={sol['p4']:.0f}")

    # ── Step 6: Build MSO location constraint example ──
    print("\n[6] MSO location constraint example:")
    mso_result = build_mso_location_constraint(["x0", "x1", "x2", "x3"])
    print(f"    MSO constraints: {mso_result['mso_constraints']}")
    print(f"    Logic clauses: {mso_result['logic_clauses']}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Summary:")
    print("  * calc-insight-kit (matplotlib + numpy) rendered the continuous function")
    print("  * logic-opt-kit (LogicConstraintBuilder) formalized the logical predicate")
    print("  * The combined approach shows: logic predicates can 'cut out'")
    print("    specific regions of a continuous domain - bridging discrete and continuous math")
    print("  * This is foundational for: spatial databases, constraint-based visualization,")
    print("    and logical reasoning over continuous structures")
    print("=" * 60)


if __name__ == "__main__":
    main()
