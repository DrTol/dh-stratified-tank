# -*- coding: utf-8 -*-
"""
Developed on Sat Aug 30 04:42:12 2025

@author: Dr. Hakan İbrahim Tol

Reference: Kleinbach EM. Performance Study of One-Dimensional Models
for Stratified Thermal Storage Tank, M.Sc. (1990) University of Wisconsin-Madison
    
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class TankParams:
    """
    Geometry, fluid properties, and loss parameters for a vertical cylindrical tank.
    Units:
      H [m], A [m^2], Nz [-]: tank height, cross-section area, number of axial cells
      rho [kg/m^3], cp [J/(kg*K)]: fluid properties (assumed constant)
      UA_side_per_m [W/(m*K)]: distributed side loss UA per meter height
      UA_top [W/K], UA_bot [W/K]: lumped top/bottom UAs
    """
    H: float
    A: float
    Nz: int
    rho: float
    cp: float
    UA_side_per_m: float
    UA_top: float
    UA_bot: float


def run_plug_flow(
    time_s: np.ndarray,          # (Nt+1,) monotonically increasing [s]
    m_dot: np.ndarray,           # (Nt,) +charge (inlet TOP, outlet BOTTOM), -discharge (inlet BOTTOM, outlet TOP), 0 idle [kg/s]
    T_in: np.ndarray,            # (Nt,) inlet temperature [°C or K]
    T_amb: np.ndarray | float,   # (Nt,) or scalar ambient temperature for losses
    T_init: np.ndarray,          # (Nz,) initial profile bottom->top
    params: TankParams,
) -> Dict[str, Any]:
    """
    Plug-flow (piston) model with semi-Lagrangian advection and Strang-split heat loss.

    Returns dict:
      - 'T_hist': (Nt+1, Nz) temperature history bottom->top
      - 'T_return': (Nt,) outlet temperature seen by the DH network each step
      - 'Q_loss_W': (Nt,) average heat-loss power to ambient over each step [W]
      - 'meta': {'dz': float, 'ports_per_step': list[(inlet_str, outlet_str)]}

    Notes on sign convention:
      m_dot[k] > 0  → charging, inlet at TOP, outlet at BOTTOM, downward internal velocity
      m_dot[k] < 0  → discharging, inlet at BOTTOM, outlet at TOP, upward internal velocity
      m_dot[k] = 0  → idle (no flow); T_return[k] = NaN
    """
    # --- input checks / shaping ---
    time_s = np.asarray(time_s, dtype=float)
    m_dot   = np.asarray(m_dot, dtype=float)
    T_in    = np.asarray(T_in, dtype=float)
    T       = np.asarray(T_init, dtype=float).copy()

    Nt = m_dot.size
    if time_s.size != Nt + 1:
        raise ValueError("time_s must have length Nt+1 (boundaries of Nt steps).")
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

    # --- grid ---
    dz = params.H / params.Nz
    T_hist   = np.zeros((Nt + 1, params.Nz), dtype=float)
    T_return = np.full(Nt, np.nan, dtype=float)
    Q_loss_W = np.zeros(Nt, dtype=float)
    T_hist[0] = T

    ports_log: list[tuple[str, str]] = []

    # --- time loop ---
    for k in range(Nt):
        dt = time_s[k+1] - time_s[k]
        if dt <= 0:
            raise ValueError("time_s must be strictly increasing.")
        md   = m_dot[k]
        Tin  = T_in[k]
        Tamb = T_amb[k]

        # port selection & velocity
        if md > 0:  # charging
            inlet, outlet = "top", "bottom"
            v = -(md / (params.rho * params.A))  # negative = downward
            outlet_idx = 0
        elif md < 0:  # discharging
            inlet, outlet = "bottom", "top"
            v = +(abs(md) / (params.rho * params.A))  # positive = upward
            outlet_idx = params.Nz - 1
        else:
            inlet, outlet = "idle", "idle"
            v = 0.0
            outlet_idx = None
        ports_log.append((inlet, outlet))

        # (1) half-step heat loss (explicit on current T)
        T, E1 = _apply_heat_loss_half_step(T, Tamb, dt * 0.5, params, dz)

        # (2) return temperature (what DH sees this step)
        if outlet_idx is not None:
            T_return[k] = T[outlet_idx]
        else:
            T_return[k] = np.nan

        # (3) advection (semi-Lagrangian plug flow)
        if v != 0.0:
            T = _advect_plug_flow(T, v, dt, Tin, params.H, dz)

        # (4) second half-step heat loss
        T, E2 = _apply_heat_loss_half_step(T, Tamb, dt * 0.5, params, dz)

        T_hist[k+1] = T
        Q_loss_W[k] = (E1 + E2) / dt  # average loss power over step [W]

    return {
        "T_hist": T_hist,
        "T_return": T_return,
        "Q_loss_W": Q_loss_W,
        "meta": {"dz": dz, "ports_per_step": ports_log},
    }


# ======================= internals =======================

def _advect_plug_flow(
    T_old: np.ndarray, v: float, dt: float, T_in: float, H: float, dz: float
) -> np.ndarray:
    """
    Semi-Lagrangian advection: T_new(z) = T_old(z - v*dt); inflow boundary set to T_in.
    Grid is cell-centered: z_i = (i+0.5)*dz, i=0..Nz-1 (bottom->top).
    v>0 upward; v<0 downward.
    """
    Nz = T_old.size
    T_new = np.empty_like(T_old)

    z_centers = (np.arange(Nz) + 0.5) * dz
    z0 = z_centers - v * dt  # departure point

    # fractional cell index on old grid: i = z/dz - 0.5
    idx_f = z0 / dz - 0.5

    for i in range(Nz):
        ii = idx_f[i]
        if ii <= -1.0:
            # came from below bottom (upward inflow from bottom)
            T_new[i] = T_in
        elif ii >= Nz:
            # came from above top (downward inflow from top)
            T_new[i] = T_in
        else:
            i0 = int(np.floor(ii))
            alpha = ii - i0
            if i0 < 0:
                # blend of inflow (bottom) and first cell
                alpha = max(alpha, 0.0)
                T_new[i] = (1.0 - alpha) * T_in + alpha * T_old[0]
            elif i0 >= Nz - 1:
                # blend of last cell and inflow (top)
                alpha = min(max(alpha, 0.0), 1.0)
                T_new[i] = (1.0 - alpha) * T_old[-1] + alpha * T_in
            else:
                # linear interpolation inside domain
                T_new[i] = (1.0 - alpha) * T_old[i0] + alpha * T_old[i0 + 1]
    return T_new


def _apply_heat_loss_half_step(
    T: np.ndarray, T_amb: float, dt_half: float, p: TankParams, dz: float
) -> Tuple[np.ndarray, float]:
    """
    Explicit half-step heat loss:
      - distributed side loss per cell via UA_side_per_m
      - lumped top/bottom losses applied to end cells
    Returns (T_new, E_loss_J) with energy removed over this half-step.
    """
    UA_side_cell = p.UA_side_per_m * dz     # [W/K] per cell
    m_cell = p.rho * p.A * dz               # [kg]
    C_cell = m_cell * p.cp                  # [J/K]

    # side losses for each cell
    dQ_side_W = UA_side_cell * (T - T_amb)        # [W] per cell
    E_side = dQ_side_W * dt_half                  # [J] per cell
    T_new = T - E_side / C_cell

    E_top = 0.0
    E_bot = 0.0

    # top loss
    if p.UA_top > 0.0:
        dQ_top_W = p.UA_top * (T_new[-1] - T_amb)
        E_top = dQ_top_W * dt_half
        T_new[-1] -= E_top / C_cell

    # bottom loss
    if p.UA_bot > 0.0:
        dQ_bot_W = p.UA_bot * (T_new[0] - T_amb)
        E_bot = dQ_bot_W * dt_half
        T_new[0] -= E_bot / C_cell

    E_total = float(np.sum(E_side)) + float(E_top) + float(E_bot)
    return T_new, E_total

