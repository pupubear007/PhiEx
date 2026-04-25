"""
engine.py  —  ASE-style adapter skeleton for the reaction workbench.

Framework mapping (Wang 2025):
    ϕ  (theory)      → Calculator:   defines V(state) and forces
    ∃  (existence)   → State:         atomic positions / reaction-coordinate value
    s∃ (study)       → Trajectory:    a sampled sequence of States
    tϕ (refined)     → FittedTheory:  parameters inferred from many Trajectories
    i  (iteration)   → BatchRunner:   drives s∃ ⟳ tϕ loops
    ↑↓ (polarity)    → direction flag on integrators

This file is ~150 lines on purpose. The discipline to learn is ASE's:
every backend (PySCF, OpenMM, a toy model, an ML potential) wears the
same Calculator coat and is swappable.

Run the toy demo at the bottom:   python engine.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Protocol
import math
import random


# ═══════════════════════════════════════════════════════════════════════════
# 1.  STATE  ( ∃ )  — plain data, no physics.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class State:
    """A point in configuration space. For the real tool this becomes an
    Atoms-like object with positions, velocities, cell, etc. For the F+H₂
    toy it's just a 1-D reaction coordinate ξ with velocity."""
    xi: float               # reaction coordinate in [0, 1]
    v:  float               # velocity along ξ
    t:  float = 0.0         # elapsed time (fs-ish)
    phase: str = "reactant" # classification tag


# ═══════════════════════════════════════════════════════════════════════════
# 2.  CALCULATOR  ( ϕ )  — the theory. Swap this and you swap physics.
# ═══════════════════════════════════════════════════════════════════════════

class Calculator(Protocol):
    """ASE-style interface. Every physics backend implements this."""
    def energy(self, state: State) -> float: ...
    def force(self, state: State)  -> float: ...


@dataclass
class ToyBarrier:
    """Eckart-like 1-D barrier along the reaction coordinate.
    The same model the HTML prototype uses. This stands in for what will
    later be PySCFCalculator, OpenMMCalculator, MACECalculator, etc."""
    Ea: float = 1.60        # barrier height
    dE: float = -1.40       # exoergicity

    def energy(self, s: State) -> float:
        barrier = self.Ea * math.exp(-((s.xi - 0.5) / 0.18) ** 2)
        slope   = self.dE * (1 / (1 + math.exp(-10 * (s.xi - 0.5))))
        return barrier + slope

    def force(self, s: State) -> float:
        eps = 1e-4
        up   = State(s.xi + eps, s.v); dn = State(s.xi - eps, s.v)
        return -(self.energy(up) - self.energy(dn)) / (2 * eps)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  INTEGRATOR  —  propagates  ∃  forward under  ϕ .  Verlet, m = 1.
# ═══════════════════════════════════════════════════════════════════════════

def propagate(s: State, calc: Calculator, dt: float = 0.02) -> State:
    f = calc.force(s)
    v_new  = s.v  + f * dt
    xi_new = s.xi + v_new * dt
    phase = ("reactant"   if xi_new < 0.35 else
             "product"    if xi_new > 0.65 else
             "transition")
    return State(xi=xi_new, v=v_new, t=s.t + dt, phase=phase)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TRAJECTORY  ( s∃ )  —  a single ϕ → ∃ run.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    samples: list[State] = field(default_factory=list)
    reacted: bool = False

    def run(self, initial: State, calc: Calculator, n_max: int = 600) -> "Trajectory":
        s = initial
        self.samples.append(s)
        for _ in range(n_max):
            s = propagate(s, calc)
            self.samples.append(s)
            if s.xi > 1.05:   self.reacted = True;  break
            if s.xi < -0.05:  self.reacted = False; break
        else:
            self.reacted = s.xi > 0.65
        return self


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BATCH  ( i : s∃ ⟳ tϕ )  —  inductive refinement loop.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FittedTheory:
    """The tϕ object: parameters we estimate from many trajectories."""
    fitted_Ea: float | None = None
    n_trials: int = 0
    n_reactions: int = 0
    def P(self) -> float:
        return self.n_reactions / self.n_trials if self.n_trials else 0.0


def run_batch(calc: Calculator, n: int = 200, Ek_range=(0.1, 4.0)) -> FittedTheory:
    theory = FittedTheory()
    bins: dict[int, list[int]] = {}  # idx -> [trials, reactions]
    for _ in range(n):
        Ek = random.uniform(*Ek_range)
        s0 = State(xi=0.02, v=math.sqrt(2 * Ek))
        traj = Trajectory().run(s0, calc)
        theory.n_trials += 1
        theory.n_reactions += int(traj.reacted)
        b = int(Ek / 0.4)
        bins.setdefault(b, [0, 0])
        bins[b][0] += 1
        bins[b][1] += int(traj.reacted)

    # crude threshold fit: lowest bin with P ≥ 0.3  ≈  effective Ea
    for b in sorted(bins):
        trials, reactions = bins[b]
        if trials and reactions / trials >= 0.3:
            theory.fitted_Ea = (b + 0.5) * 0.4
            break
    return theory


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ϕ → ∃ reaction workbench — Python adapter demo\n")

    calc = ToyBarrier(Ea=1.60, dE=-1.40)
    print(f"theory ϕ: ToyBarrier(Ea={calc.Ea}, ΔE={calc.dE})\n")

    # ── single deductive run ϕ → ∃ ──
    print("── ϕ → ∃ single trajectory ──")
    s0 = State(xi=0.02, v=math.sqrt(2 * 2.0))   # Ek = 2.0
    traj = Trajectory().run(s0, calc)
    last = traj.samples[-1]
    print(f"  Ek=2.00 → final ξ={last.xi:.3f}  reacted={traj.reacted}\n")

    # ── inductive batch s∃ ⟳ tϕ ──
    print("── s∃ ⟳ tϕ  inductive batch (i=1) ──")
    t_phi = run_batch(calc, n=500)
    print(f"  trials     = {t_phi.n_trials}")
    print(f"  reactions  = {t_phi.n_reactions}")
    print(f"  P(react)   = {t_phi.P():.3f}")
    print(f"  fitted Ea  = {t_phi.fitted_Ea}  (true Ea = {calc.Ea})")
    print("\n  → swap ToyBarrier for PySCFCalculator to go quantum.")
    print("  → swap for OpenMMCalculator to go biomolecular.")
    print("  → the Trajectory / Batch machinery above does not change.")
