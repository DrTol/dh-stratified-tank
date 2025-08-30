# -*- coding: utf-8 -*-
"""
Developed on Sat Aug 30 06:06:26 2025

@author: Dr. Hakan İbrahim Tol

"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class TankParams:
    H: float
    A: float
    Nz: int
    rho: float
    cp: float
    UA_side_per_m: float
    UA_top: float    # north
    UA_bot: float    # south

def run_fv_cn(
    time_s: np.ndarray,
    m_dot: np.ndarray,
    T_in: np.ndarray,
    T_amb: np.ndarray | float,
    T_init: np.ndarray,
    params: TankParams,
    *,
    theta: float = 0.5,     # 0=explicit, 0.5=CN, 1=implicit
    D_ax: float = 1e-4,     # [m^2/s] axial dispersion
    safety: float = 0.9,    # internal substepping margin
    fo_cap: float = 1.0     # Fo cap for accuracy
) -> Dict[str, Any]:

    time_s = np.asarray(time_s, dtype=float)
    m_dot  = np.asarray(m_dot, dtype=float)
    T_in   = np.asarray(T_in, dtype=float)
    T      = np.asarray(T_init, dtype=float).copy()

    Nt = m_dot.size
    if time_s.size != Nt + 1: raise ValueError("time_s must have length Nt+1.")
    if T.size != params.Nz:   raise ValueError("T_init length must equal Nz.")
    if T_in.size != Nt:       raise ValueError("T_in must have length Nt.")
    if np.isscalar(T_amb):
        T_amb = np.full(Nt, float(T_amb), dtype=float)
    else:
        T_amb = np.asarray(T_amb, dtype=float)
        if T_amb.size != Nt: raise ValueError("T_amb must be scalar or length Nt.")

    Nz = params.Nz
    dz = params.H / Nz
    C_cell = params.rho * params.A * dz * params.cp

    H_side = params.UA_side_per_m * dz
    H_diag = np.full(Nz, H_side, dtype=float)
    H_diag[0]  += params.UA_bot   # south
    H_diag[-1] += params.UA_top   # north
    S_max = np.max(H_diag) / C_cell if theta < 1.0 else 0.0

    T_hist   = np.zeros((Nt + 1, Nz), dtype=float)
    T_return = np.full(Nt, np.nan, dtype=float)
    Q_loss_W = np.zeros(Nt, dtype=float)
    T_hist[0] = T

    ports_log: list[tuple[str, str]] = []

    for k in range(Nt):
        dt_full = time_s[k+1] - time_s[k]
        if dt_full <= 0: raise ValueError("time_s must be strictly increasing.")
        md   = m_dot[k]
        Tin  = T_in[k]
        Tamb = T_amb[k]

        # sign convention: +m_dot = charging (inlet north, outlet south)
        if md > 0:
            inlet, outlet = "north", "south"
            u = -(md / (params.rho * params.A))   # downward = negative
            out_idx = 0                           # south
        elif md < 0:
            inlet, outlet = "south", "north"
            u = +(abs(md) / (params.rho * params.A))  # upward = positive
            out_idx = Nz - 1                        # north
        else:
            inlet, outlet = "idle", "idle"
            u = 0.0
            out_idx = None
        ports_log.append((inlet, outlet))

        if out_idx is not None:
            T_return[k] = T[out_idx]

        aL, aD, aU, src_adv = _build_fv_upwind_operators(Nz, dz, u, D_ax, C_cell, Tin)

        dt_left = dt_full
        E_loss_sum = 0.0
        while dt_left > 0.0:
            dt_lim = _safe_dt(u, dz, D_ax, S_max, theta, fo_cap)
            dt = min(dt_left, safety * dt_lim)

            A_main, A_lower, A_upper, rhs = _assemble_theta_system(
                T, H_diag, aL, aD, aU, src_adv, C_cell, dt, theta, Tamb
            )
            T_new = _solve_tridiag(A_lower, A_main, A_upper, rhs)

            E_loss_sum += float(np.sum(H_diag * (0.5*(T + T_new) - Tamb))) * dt
            T = T_new
            dt_left -= dt

        T_hist[k+1] = T
        Q_loss_W[k] = E_loss_sum / dt_full

    return {
        "T_hist": T_hist,
        "T_return": T_return,
        "Q_loss_W": Q_loss_W,
        "meta": {"dz": dz, "ports_per_step": ports_log},
    }

# ---------- internals ----------

def _safe_dt(u: float, dz: float, D_ax: float, S_max: float, theta: float, fo_cap: float) -> float:
    eps = 1e-12
    dt_cfl  = dz / max(abs(u), eps)
    dt_fo   = (fo_cap * dz**2 / max(D_ax, eps)) if D_ax > 0.0 else 1e30
    dt_sink = (1.0 / ((1.0 - theta) * S_max)) if (theta < 1.0 and S_max > 0.0) else 1e30
    return min(dt_cfl, dt_fo, dt_sink)

def _build_fv_upwind_operators(
    Nz: int, dz: float, u: float, D_ax: float, C_cell: float, T_in: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aL = np.zeros(Nz)      # south neighbor (i-1)
    aD = np.zeros(Nz)      # cell i
    aU = np.zeros(Nz)      # north neighbor (i+1)
    src_adv = np.zeros(Nz)

    # diffusion (central), zero-gradient at south/north
    K = C_cell * max(D_ax, 0.0) / dz**2
    if K > 0.0:
        aL[1:] += K
        aD[:]  += -2.0 * K
        aU[:-1]+= K
        aD[0]  += K          # Neumann at south
        aD[-1] += K          # Neumann at north

    # advection (upwind)
    adv = C_cell * abs(u) / dz
    if u > 0.0:  # flow northwards (up)
        aD[:]      += -adv
        aL[1:]     += +adv
        src_adv[0] += adv * T_in      # inflow from south boundary
    elif u < 0.0:  # flow southwards (down)
        aD[:]       += -adv
        aU[:-1]     += +adv
        src_adv[-1] += adv * T_in     # inflow from north boundary
    # outflow handled by main diagonal -adv at the outlet cell

    return aL, aD, aU, src_adv

def _assemble_theta_system(
    Tn: np.ndarray,
    H_diag: np.ndarray,
    L_lower: np.ndarray, L_main: np.ndarray, L_upper: np.ndarray,
    src_adv: np.ndarray,
    C_cell: float, dt: float, theta: float, T_amb_k: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M_lower = L_lower / C_cell
    M_main  = L_main  / C_cell - H_diag / C_cell
    M_upper = L_upper / C_cell
    b_vec   = (H_diag * T_amb_k + src_adv) / C_cell

    A_lower = -theta * dt * M_lower
    A_main  = 1.0   - theta * dt * M_main
    A_upper = -theta * dt * M_upper

    B_lower = (1.0 - theta) * dt * M_lower
    B_main  = 1.0   + (1.0 - theta) * dt * M_main
    B_upper = (1.0 - theta) * dt * M_upper

    Tm = np.concatenate(([Tn[0]],  Tn[:-1]))  # south neighbor (i-1)
    Tp = np.concatenate(( Tn[1:],  [Tn[-1]])) # north neighbor (i+1)

    rhs = B_lower*Tm + B_main*Tn + B_upper*Tp + dt * b_vec
    return A_main, A_lower, A_upper, rhs

def _solve_tridiag(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = b.size
    ac, bc, cc, dc = a.copy(), b.copy(), c.copy(), d.copy()
    for i in range(1, n):
        m = ac[i] / bc[i-1]
        bc[i] -= m * cc[i-1]
        dc[i] -= m * dc[i-1]
    x = np.empty(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i]*x[i+1]) / bc[i]
    return x
