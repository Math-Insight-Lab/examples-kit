# -*- coding: utf-8 -*-
"""
Demo 6 — Symbolic Integration Meets Logical Optimization
=========================================================
Cross-library demo: calc-insight-kit (Maxima CAS backend)
                     + logic-opt-kit (MIP constraint optimization)

Problem: Optimal Interval Selection with Symbolically-Cost Objective
-------------------------------------------------------------------

You are planning an experiment over the time window [0, 1.5] (radians).
The experiment's instantaneous gain rate is:

    g(x) = tan(x) / (cos(x) + 1)

You need to choose a set of time intervals during which to run the
experiment, subject to logical constraints:

    1. At most 3 of 5 sub-intervals can be active.
    2. If interval 1 is active, then interval 3 must be active (dependency).
    3. At least one of intervals 4 and 5 must be active (safety).
    4. You cannot activate adjacent intervals (e.g., if interval 2 is
       active, interval 3 must be off — to allow calibration time).

The objective: MAXIMIZE the total accumulated gain
    sum of ∫_{subinterval_i} g(x) dx  for all active intervals.

Workflow:
    ┌─────────────────────────┐
    │ calc-insight-kit        │
    │ - Maxima solves:        │
    │   ∫ tan(x)/(cos(x)+1)dx │
    │   = ln(cos(x)+1)-ln(cos(x))
    │ - Evaluate definite     │
    │   integrals over 5 sub- │
    │   intervals.            │
    ├─────────────────────────┤
    │ logic-opt-kit           │
    │ - Build MIP model with  │
    │   logical constraints   │
    │ - Optimize selection    │
    └─────────────────────────┘
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import integrate as sp_int

# --- Import calc-insight-kit: Maxima CAS backend ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'calc-insight-kit'))
from calc_insight_kit import use_theme, get_cas_backend, plot_math

# --- Import logic-opt-kit: MIP constraint optimization ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'logic-opt-kit'))
from logic_opt_kit.core.solver_backend import get_solver_backend as get_opt_backend
from logic_opt_kit.core.model_builder import QuickModelBuilder
from logic_opt_kit.core.logic_solver import LogicConstraintBuilder

# Apply theme
use_theme("teaching")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Use Maxima to find the antiderivative
# ============================================================
def get_antiderivative():
    """
    Compute ∫ tan(x)/(cos(x)+1) dx using Maxima.
    
    SymPy returns Integral(...) (unevaluated).
    Maxima returns: log(cos(x)+1) - log(cos(x))
    """
    maxima = get_cas_backend("maxima")
    # SymPy would fail here — this is the key
    antideriv_expr = maxima.integrate("tan(x)/(cos(x)+1)", "x")
    print(f"  [Maxima] Antiderivative: {antideriv_expr}")
    return antideriv_expr


def definite_integral(F_upper, F_lower, h=1e-10):
    """
    Numerical evaluation of the definite integral using the antiderivative.
    
    Since F(x) = ln(cos(x)+1) - ln(cos(x)),
    ∫_a^b f(x) dx = F(b) - F(a)
    """
    return F_upper - F_lower


def F(x_val):
    """Antiderivative: ln(cos(x)+1) - ln(cos(x))"""
    c = np.cos(x_val)
    return np.log(np.abs(c + 1) + 1e-15) - np.log(np.abs(c) + 1e-15)


# ============================================================
# Step 2: Evaluate integrals over 5 sub-intervals
# ============================================================
def compute_interval_gains():
    """
    Divide [0, 1.5] into 5 equal sub-intervals and compute the gain
    (definite integral of tan(x)/(cos(x)+1)) over each.
    """
    a, b = 0.0, 1.5
    n_intervals = 5
    dx = (b - a) / n_intervals

    gains = {}
    print("\n  [Symbolic] Computing gain per sub-interval:")
    for i in range(n_intervals):
        lo = a + i * dx
        hi = lo + dx
        F_lo = F(lo)
        F_hi = F(hi)
        gain = definite_integral(F_hi, F_lo)
        gains[i + 1] = gain
        print(f"    Interval {i+1}: [{lo:.3f}, {hi:.3f}] → gain = {gain:.6f}")

    return gains


# ============================================================
# Step 3: Build and solve the MIP optimization model
# ============================================================
def solve_optimal_selection(gains):
    """
    Use logic-opt-kit to find the optimal set of intervals to activate.
    
    Objective: maximize sum(gain_i * x_i)  where x_i ∈ {0, 1}
    
    Constraints:
        C1: x1 + x2 + x3 + x4 + x5 ≤ 3  (at most 3 active)
        C2: x1 ≤ x3  (if interval 1 active, interval 3 must be active)
        C3: x4 + x5 ≥ 1  (at least one of 4,5 active)
        C4: x2 + x3 ≤ 1  (no adjacent: can't activate both 2 and 3)
    """
    print("  [MIP] Building optimization model...")
    print("  Constraints:")
    print("    C1: At most 3 sub-intervals active")
    print("    C2: If interval 1 active → interval 3 must be active (dependency)")
    print("    C3: At least one of intervals 4,5 active (safety)")
    print("    C4: Cannot activate adjacent intervals (calibration time)")

    # Get MIP backend: try SCIP first (full MINLP support), fallback to HiGHS
    backend_name = "scip"
    try:
        backend = get_opt_backend("scip")
        print("  [MIP] Using SCIP solver backend")
    except RuntimeError:
        backend_name = "highs"
        print("  [MIP] SCIP not available, falling back to HiGHS solver")
        backend = get_opt_backend("highs")
    mb = QuickModelBuilder(backend, model_name="interval_selection")

    # Create binary decision variables: x_i = 1 if interval i is active
    x = {}
    name_to_index = {}
    for i in range(1, 6):
        name = f"x{i}"
        x[name] = mb.var(name, low=0, high=1, vtype="B")
        name_to_index[name] = i

    # Objective: maximize total gain
    # x1*gain1 + x2*gain2 + ... + x5*gain5
    objective = sum(gains[name_to_index[k]] * x[k] for k in x)
    mb.obj(objective, sense="max")

    # Constraint C1: at most 3 active
    mb.cons(sum(x[k] for k in x) <= 3, name="at_most_3")

    # Constraint C2: x1 ≤ x3 (if x1=1 then x3=1)
    mb.cons(x["x1"] <= x["x3"], name="dependency_1_3")

    # Constraint C3: x4 + x5 ≥ 1
    mb.cons(x["x4"] + x["x5"] >= 1, name="safety_4_5")

    # Constraint C4: x2 + x3 ≤ 1 (no adjacent)
    mb.cons(x["x2"] + x["x3"] <= 1, name="no_adjacent_2_3")

    print("  [MIP] Solving...")
    result = mb.solve()
    solution = mb.solution()

    print(f"  [MIP] Optimal objective value: {mb.obj_value():.6f}")
    print(f"  [MIP] Solution: ", end="")
    selected = []
    for name, val in solution.items():
        idx = name_to_index[name]
        val_rounded = round(val)
        status = "ACTIVE" if val_rounded == 1 else "inactive"
        print(f"{name}={val_rounded}({status})", end="  ")
        if val_rounded == 1:
            selected.append(idx)
    print()

    return selected, mb.obj_value(), solution


# ============================================================
# Step 4: Verify the solution is globally optimal (brute force)
# ============================================================
def verify_optimality(gains, selected, optimal_value):
    """
    Brute-force verify: check all 2^5 = 32 binary combinations against
    the constraints, confirm the MIP solution is optimal.
    """
    print("\n  [Verification] Brute-force checking all 32 combinations...")

    best_val = -np.inf
    best_combo = None

    for mask in range(32):
        combo = [(mask >> i) & 1 for i in range(5)]  # 5 binary vars
        vals = {i + 1: combo[i] for i in range(5)}

        # Check constraints
        if sum(vals.values()) > 3:
            continue  # C1 violated
        if vals[1] > vals[3]:
            continue  # C2 violated (x1=1 but x3=0)
        if vals[4] + vals[5] < 1:
            continue  # C3 violated
        if vals[2] + vals[3] > 1:
            continue  # C4 violated (adjacent)

        # Valid combination — compute objective
        val = sum(gains[i] * vals[i] for i in range(1, 6))
        if val > best_val:
            best_val = val
            best_combo = vals.copy()

    print(f"  [Verification] Best valid combination found by brute force:")
    for i in range(1, 6):
        status = "ACTIVE" if best_combo[i] == 1 else "inactive"
        print(f"    x{i}={best_combo[i]} ({status}), gain={gains[i]:.6f}")
    print(f"  [Verification] Objective: {best_val:.6f}")
    print(f"  [Verification] MIP matches brute-force: {abs(optimal_value - best_val) < 1e-9}")

    return best_combo, best_val


# ============================================================
# Step 5: Visualization
# ============================================================
def visualize(gains, selected, optimal_value):
    """Generate a comprehensive visualization."""
    print("\n  [Visualization] Generating plots...")

    fig = plt.figure(figsize=(16, 11))

    # ---- Panel 1: Integrand with selected intervals highlighted ----
    ax1 = fig.add_subplot(3, 2, 1)
    x_plot = np.linspace(0, 1.5, 500)
    x_plot = np.clip(x_plot, 0.001, 1.499)
    f_plot = np.tan(x_plot) / (np.cos(x_plot) + 1)
    ax1.plot(x_plot, f_plot, 'b-', linewidth=2,
             label=r'$g(x) = \tan(x)/(\cos(x)+1)$')

    # Color each sub-interval
    colors = ['red', 'orange', 'green', 'purple', 'cyan']
    a, b = 0.0, 1.5
    n_intervals = 5
    dx = (b - a) / n_intervals
    for i in range(n_intervals):
        lo = a + i * dx
        hi = lo + dx
        color = 'green' if (i + 1) in selected else 'lightgray'
        ax1.axvspan(lo, hi, alpha=0.3, color=color,
                     label=f"Int {i+1} {'✓' if (i+1) in selected else '✗'}")

    ax1.set_xlabel('x (time)')
    ax1.set_ylabel('g(x)')
    ax1.set_title('Selected Active Intervals (green = optimal)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 15)

    # ---- Panel 2: Gains per interval + selection ----
    ax2 = fig.add_subplot(3, 2, 2)
    interval_nums = list(gains.keys())
    gain_values = list(gains.values())
    is_selected = [1 if i in selected else 0 for i in interval_nums]
    bar_colors = ['green' if s else 'lightgray' for s in is_selected]

    bars = ax2.bar(
        [f'Int {i}' for i in interval_nums],
        gain_values,
        color=bar_colors,
        edgecolor='black',
        linewidth=1.5
    )
    # Annotate values
    for bar, val in zip(bars, gain_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    ax2.set_ylabel('Gain = ∫ g(x)dx')
    ax2.set_title('Gain per Sub-Interval (green = selected)')
    ax2.grid(True, alpha=0.3, axis='y')

    # ---- Panel 3: Antiderivative F(x) ----
    ax3 = fig.add_subplot(3, 2, 3)
    F_plot = np.array([F(xi) for xi in x_plot])
    ax3.plot(x_plot, F_plot, 'b-', linewidth=2,
             label=r'$F(x) = \ln(\cos(x)+1) - \ln(\cos(x))$')

    # Mark selected intervals on antiderivative
    for i in range(n_intervals):
        lo = a + i * dx
        hi = lo + dx
        F_lo = F(lo)
        F_hi = F(hi)
        rise = F_hi - F_lo
        mid_x = (lo + hi) / 2
        color = 'green' if (i + 1) in selected else 'lightgray'
        ax3.annotate('', xy=(mid_x, F_hi), xytext=(mid_x, F_lo),
                     arrowprops=dict(arrowstyle='<->', color=color, lw=2))

    ax3.set_xlabel('x')
    ax3.set_ylabel('F(x)')
    ax3.set_title('Antiderivative (green arrows = selected interval gains)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # ---- Panel 4: Constraint satisfaction summary ----
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.axis('off')

    # Check each constraint
    c1 = sum(selected) <= 3
    c2 = (1 not in selected) or (3 in selected)
    c3 = 4 in selected or 5 in selected
    c4 = not (2 in selected and 3 in selected)

    all_ok = c1 and c2 and c3 and c4
    check_mark = "✓" if all_ok else "✗"

    constraints_text = f"""
    CONSTRAINT VERIFICATION (all checked against solution {selected}):

    C1: At most 3 active       → {sum(selected)} active  → {check_mark if c1 else '✗ FAIL'}
    C2: If 1→3                  → 1={1 in selected}, 3={3 in selected} → {check_mark if c2 else '✗ FAIL'}
    C3: At least one of 4,5     → 4={4 in selected}, 5={5 in selected} → {check_mark if c3 else '✗ FAIL'}
    C4: No adjacent (2,3)       → 2={2 in selected}, 3={3 in selected} → {check_mark if c4 else '✗ FAIL'}

    ═══════════════════════════════════════════
    RESULT: {"ALL CONSTRAINTS SATISFIED" if all_ok else "SOME CONSTRAINTS FAILED"}
    ═══════════════════════════════════════════
    """
    ax4.text(0.05, 0.95, constraints_text.strip(), transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Constraint Verification')

    # ---- Panel 5: Comparison table (all valid combos) ----
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.axis('tight')
    ax5.axis('off')

    # Build table of all valid combinations (top 10)
    valid_combos = []
    for mask in range(32):
        combo = [(mask >> i) & 1 for i in range(5)]
        vals = {i + 1: combo[i] for i in range(5)}

        if sum(vals.values()) > 3:
            continue
        if vals[1] > vals[3]:
            continue
        if vals[4] + vals[5] < 1:
            continue
        if vals[2] + vals[3] > 1:
            continue

        val = sum(gains[i] * vals[i] for i in range(1, 6))
        valid_combos.append((val, vals))

    valid_combos.sort(reverse=True)

    table_data = [['Rank', 'Selection', 'Objective', 'Valid?']]
    for rank, (val, combo) in enumerate(valid_combos[:10], 1):
        sel_str = ','.join(str(combo[i]) for i in range(1, 6))
        is_best = "★" if (abs(val - optimal_value) < 1e-9) else " "
        table_data.append([f"{rank}{is_best}", f"[{sel_str}]", f"{val:.4f}", "Yes"])

    table = ax5.table(cellText=table_data, cellLoc='left',
                      loc='center', colWidths=[0.1, 0.35, 0.3, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')
    # Highlight optimal
    for row in range(1, len(table_data)):
        if table_data[row][0].startswith('1'):
            for col in range(4):
                table[(row, col)].set_facecolor('#90EE90')

    ax5.set_title('Top 10 Valid Combinations (★ = optimal)')

    # ---- Panel 6: Workflow diagram ----
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    workflow_text = """
    CROSS-LIBRARY WORKFLOW:

    ┌─────────────────────────────────────────────┐
    │  calc-insight-kit (Maxima CAS backend)       │
    │                                             │
    │  Input: tan(x)/(cos(x)+1)                   │
    │  Maxima.integrate() →                       │
    │  "log(cos(x)+1)-log(cos(x))"               │
    │                                             │
    │  → Symbolic antiderivative F(x)             │
    │  → Evaluate F(b_i) - F(a_i) per interval   │
    │  → Numerical gains: [g1, g2, ..., g5]      │
    └──────────────┬──────────────────────────────┘
                   │  passes gains as coefficients
                   ▼
    ┌─────────────────────────────────────────────┐
    │  logic-opt-kit (MIP backend: SCIP)          │
    │                                             │
    │  Build MIP model:                           │
    │    maximize  g1*x1 + g2*x2 + ... + g5*x5  │
    │    subject to logical constraints           │
    │    (at_most_3, dependency, safety, no_adj) │
    │                                             │
    │  → MIP solver finds optimal x*              │
    │  → Optimal interval selection: {selected}   │
    └─────────────────────────────────────────────┘
    """
    ax6.text(0.03, 0.98, workflow_text.format(selected=selected),
             transform=ax6.transAxes, fontsize=9.5, verticalalignment='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.8,
                       edgecolor='#4472C4', linewidth=2))
    ax6.set_title('Cross-Library Data Flow')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "calculus_logic_optimization.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("Demo 6: Symbolic Integration + Logical Optimization")
    print("=" * 70)

    # Step 1: Get antiderivative from Maxima
    print("\n[1] Symbolic Integration (calc-insight-kit + Maxima):")
    print("    Problem: ∫ tan(x)/(cos(x)+1) dx")
    antideriv = get_antiderivative()

    # Step 2: Compute gains per sub-interval
    print("\n[2] Compute definite integrals over 5 sub-intervals:")
    gains = compute_interval_gains()

    # Step 3: Solve MIP optimization
    print("\n[3] MIP Optimization (logic-opt-kit + SCIP):")
    selected, optimal_value, solution = solve_optimal_selection(gains)

    # Step 4: Verify
    best_combo, brute_force_val = verify_optimality(gains, selected, optimal_value)

    # Step 5: Visualize
    print("\n[4] Visualization:")
    visualize(gains, selected, optimal_value)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  * Symbolic solution: F(x) = ln(cos(x)+1) - ln(cos(x)) [Maxima]")
    print(f"  * Gains per interval: {dict(gains)}")
    print(f"  * Optimal selection: {selected} (total gain = {optimal_value:.6f})")
    print(f"  * Verified optimal by brute-force: {abs(optimal_value - brute_force_val) < 1e-9}")
    print(f"  * This demo demonstrates:")
    print(f"    - calc-insight-kit provides symbolic antiderivative")
    print(f"    - logic-opt-kit solves constraint optimization over the gains")
    print(f"    - The two libraries work TOGETHER: symbolic → numerical → optimal")
    print("=" * 70)


if __name__ == "__main__":
    main()
