# -*- coding: utf-8 -*-
"""
Developed on Sat Aug 30 04:53:13 2025

@author: Dr. Hakan İbrahim Tol

Reference: Kleinbach EM. Performance Study of One-Dimensional Models
for Stratified Thermal Storage Tank, M.Sc. (1990) University of Wisconsin-Madison

"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np

@dataclass
class TankParams:
    H: float
    A: float
    Nz: int
    rho: float
    cp: float
    UA_side_per_m: float
    UA_top: float
    UA_bot: float

def run_multi_node(
    time_s: np.ndarray,
    m_dot: np.ndarray,
    T_in: np.ndarray,
    T_amb: np.ndarray | float,
    T_init: np.ndarray,
    params: TankParams,
) -> Dict[str, Any]:
    time_s = np.asarray(time_s, dtype=float)
    m_dot = np.asarray(m_dot, dtype=float)
    T_in = np.asarray(T_in, dtype=float)
    T = np.asarray(T_init, dtype=float).copy()
    Nt = m_dot.size
    if time_s.size != Nt + 1:
        raise ValueError("time_s must have length Nt+1.")
    if T.size != params.Nz:
        raise ValueError("T_init length must equal Nz.")
    if T_in.size != Nt:
        raise ValueError("T_in must have length Nt.")
    if np.isscalar(T_amb):
        T_amb = np.full(Nt, float(T_amb), dtype=float)
    else:
        T_amb = np.asarray(T_amb, dtype=float)
        if T_amb.size != Nt:
            raise ValueError("T_amb must be scalar or length Nt.")

    dz = params.H / params.Nz
    m_cell = params.rho * params.A * dz

    T_hist = np.zeros((Nt + 1, params.Nz), dtype=float)
    T_return = np.full(Nt, np.nan, dtype=float)
    Q_loss_W = np.zeros(Nt, dtype=float)
    T_hist[0] = T

    ports_log: list[tuple[str, str]] = []

    for k in range(Nt):
        dt = time_s[k+1] - time_s[k]
        if dt <= 0:
            raise ValueError("time_s must be strictly increasing.")
        md = m_dot[k]
        Tin = T_in[k]
        Tamb = T_amb[k]

        if md > 0:
            inlet, outlet = "top", "bottom"
            outlet_idx = 0
        elif md < 0:
            inlet, outlet = "bottom", "top"
            outlet_idx = params.Nz - 1
        else:
            inlet, outlet = "idle", "idle"
            outlet_idx = None
        ports_log.append((inlet, outlet))

        T, E1 = _apply_heat_loss_half_step(T, Tamb, dt * 0.5, params, dz)

        if outlet_idx is not None:
            T_return[k] = T[outlet_idx]
        else:
            T_return[k] = np.nan

        if md != 0.0:
            frac = min(1.0, (abs(md) * dt) / m_cell)
            alpha = (abs(md) * dt) / (m_cell + abs(md) * dt)
            if md > 0:
                T[-1] = (1.0 - alpha) * T[-1] + alpha * Tin
                for i in range(params.Nz - 2, -1, -1):
                    T[i] = (1.0 - frac) * T[i] + frac * T[i + 1]
            else:
                T[0] = (1.0 - alpha) * T[0] + alpha * Tin
                for i in range(1, params.Nz):
                    T[i] = (1.0 - frac) * T[i] + frac * T[i - 1]

        T, E2 = _apply_heat_loss_half_step(T, Tamb, dt * 0.5, params, dz)

        T_hist[k+1] = T
        Q_loss_W[k] = (E1 + E2) / dt

    return {
        "T_hist": T_hist,
        "T_return": T_return,
        "Q_loss_W": Q_loss_W,
        "meta": {"dz": dz, "ports_per_step": ports_log},
    }

def _apply_heat_loss_half_step(
    T: np.ndarray, T_amb: float, dt_half: float, p: TankParams, dz: float
) -> Tuple[np.ndarray, float]:
    UA_side_cell = p.UA_side_per_m * dz
    m_cell = p.rho * p.A * dz
    C_cell = m_cell * p.cp

    dQ_side_W = UA_side_cell * (T - T_amb)
    E_side = dQ_side_W * dt_half
    T_new = T - E_side / C_cell

    E_top = 0.0
    E_bot = 0.0
    if p.UA_top > 0.0:
        dQ_top_W = p.UA_top * (T_new[-1] - T_amb)
        E_top = dQ_top_W * dt_half
        T_new[-1] -= E_top / C_cell
    if p.UA_bot > 0.0:
        dQ_bot_W = p.UA_bot * (T_new[0] - T_amb)
        E_bot = dQ_bot_W * dt_half
        T_new[0] -= E_bot / C_cell

    E_total = float(np.sum(E_side)) + float(E_top) + float(E_bot)
    return T_new, E_total
