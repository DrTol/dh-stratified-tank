# -*- coding: utf-8 -*-
"""
Developed on Sat Aug 30 04:44:00 2025

@author: Dr. Hakan İbrahim Tol
"""

import numpy as np
from plugflow import TankParams, run_plug_flow
from multinode import TankParams, run_multi_node
from fd_cn import TankParams, run_fd_cn
from fv_cn import TankParams, run_fv_cn

# ==== USER SETTINGS ===========================================================
# Tank geometry & fluid
H_m   = 6.0                 # [m] tank height
D_m   = 2.0                 # [m] diameter
A_m2  = 0.25 * np.pi * D_m**2
Nz    = 40                  # [-] number of axial cells (bottom->top)

rho   = 985.0               # [kg/m^3] water density
cp    = 4180.0              # [J/(kg*K)] water specific heat

# Heat losses (adjust per insulation quality)
UA_side_per_m = 3.0         # [W/(m*K)] distributed side UA per meter height
UA_top        = 4.0         # [W/K]
UA_bot        = 6.0         # [W/K]

# Time grid
t_end_min = 120.0           # [min] total simulated time
dt_s      = 10.0            # [s]   time step
time_s    = np.arange(0.0, t_end_min*60.0 + dt_s, dt_s)   # (Nt+1,)

# Operation schedule (default: CHARGING only)
m_dot_ch  = 12               # [kg/s] + => charging (inlet TOP, outlet BOTTOM)
T_in_C    = 80.0            # [°C] inlet temperature
T_amb_C   = 20.0            # [°C] ambient around tank

# Initial temperature profile
T_init_C  = 40.0            # [°C] uniform initial tank temperature
# ==============================================================================

def build_schedules(time_s, dt_s, mode="charging_only"):
    """
    Returns (m_dot, T_in, T_amb) arrays of length Nt for the chosen mode.
    Available modes:
      - 'charging_only' (default): constant +m_dot (charging)
      - 'demo': charging → idle → discharging sequence (for testing)
    """
    Nt = time_s.size - 1

    if mode == "charging_only":
        m_dot = np.full(Nt, m_dot_ch, dtype=float)
        T_in  = np.full(Nt, T_in_C, dtype=float)
        T_amb = np.full(Nt, T_amb_C, dtype=float)
        return m_dot, T_in, T_amb

    elif mode == "demo":
        # Example: 40 min charge, 20 min idle, 60 min discharge
        t_charge = 40*60
        t_idle   = 20*60
        steps_charge = int(t_charge // dt_s)
        steps_idle   = int(t_idle   // dt_s)
        steps_dis    = (Nt - steps_charge - steps_idle)

        m_dot = np.empty(Nt, dtype=float)
        m_dot[:steps_charge] = +m_dot_ch
        m_dot[steps_charge:steps_charge+steps_idle] = 0.0
        m_dot[steps_charge+steps_idle:] = -0.6 * m_dot_ch  # smaller magnitude for discharge

        T_in  = np.full(Nt, T_in_C, dtype=float)   # used only where |m_dot|>0
        T_amb = np.full(Nt, T_amb_C, dtype=float)
        return m_dot, T_in, T_amb

    else:
        raise ValueError("Unknown mode. Use 'charging_only' or 'demo'.")


def main():
    Nt = time_s.size - 1

    # Build operation schedules (choose 'charging_only' or 'demo')
    m_dot, T_in, T_amb = build_schedules(time_s, dt_s, mode="charging_only")

    # Initial profile (bottom->top)
    T_init = np.full(Nz, T_init_C, dtype=float)

    # Pack tank parameters
    params = TankParams(
        H=H_m, A=A_m2, Nz=Nz,
        rho=rho, cp=cp,
        UA_side_per_m=UA_side_per_m,
        UA_top=UA_top, UA_bot=UA_bot
    )

    # Run the selected model (plug-flow)
    # result = run_plug_flow(
    #     time_s=time_s,
    #     m_dot=m_dot,
    #     T_in=T_in,
    #     T_amb=T_amb,
    #     T_init=T_init,
    #     params=params
    # )
    
    # result = run_multi_node(
    #     time_s=time_s,
    #     m_dot=m_dot,
    #     T_in=T_in,
    #     T_amb=T_amb,
    #     T_init=T_init,
    #     params=params
    # )
    
    result = run_fv_cn(
        time_s=time_s,
        m_dot=m_dot,
        T_in=T_in,
        T_amb=T_amb,
        T_init=T_init,
        params=params,
        theta=0.55,     # e.g., slightly implicit CN
        D_ax=1e-4,     
    )
        

    T_hist   = result["T_hist"]      # (Nt+1, Nz)
    T_return = result["T_return"]    # (Nt,)
    Q_loss_W = result["Q_loss_W"]    # (Nt,)

    # Console summary
    print(f"[OK] Plug-flow run complete. Steps={Nt}, Cells={Nz}")
    print(f"Final bottom/top T: {T_hist[-1,0]:.2f} / {T_hist[-1,-1]:.2f} °C")
    tail = max(int(600//dt_s), 1)  # last 10 minutes
    print(f"Mean return T over last 10 min: {np.nanmean(T_return[-tail:]):.2f} °C")
    print(f"Average heat loss over run: {np.mean(Q_loss_W)/1000:.2f} kW")

    # Optional quicklook plots
    try:
        import matplotlib.pyplot as plt
    
        # Helper: choose snapshot indices that track the front travel (plug-flow)
        def pick_snapshot_indices(time_s, m_dot, rho, A, H, n=5, fixed_minutes=None):
            Nt = time_s.size - 1
            if fixed_minutes is not None:
                snaps_s = [min(max(0.0, 60.0*m), time_s[-1]) for m in fixed_minutes]
            else:
                # use fractions of tank height for advection distance |v|*t ≈ [0, 0.25H, 0.5H, 0.75H, H]
                v_abs = np.mean(np.abs(m_dot)) / (rho * A)  # [m/s]; crude average is fine for plotting
                if v_abs < 1e-9:
                    snaps_s = np.linspace(0.0, time_s[-1], n)  # idle; just spread in time
                else:
                    dist_targets = np.linspace(0.0, H, n)
                    snaps_s = [min(time_s[-1], d / v_abs) for d in dist_targets]
            idxs = [int(round(s / (time_s[1]-time_s[0]))) for s in snaps_s]
            idxs = [min(max(i, 0), Nt) for i in idxs]
            # ensure unique & sorted
            return sorted(list(dict.fromkeys(idxs)))
    
        # Pick indices (either adaptive by front or specify fixed minutes)
        idxs = pick_snapshot_indices(
            time_s=time_s,
            m_dot=m_dot,
            rho=rho,
            A=A_m2,
            H=H_m,
            n=5,
            fixed_minutes=None  # e.g., [0, 5, 10, 20, 30] to force specific times
        )
    
        # 1) Return temperature
        plt.figure()
        plt.plot(time_s[:-1]/60.0, T_return)
        plt.xlabel("Time [min]"); plt.ylabel("Return temperature [°C]")
        plt.title("Tank return temperature"); plt.grid(True)
    
        # 2) Heat loss
        plt.figure()
        plt.plot(time_s[:-1]/60.0, Q_loss_W/1000.0)
        plt.xlabel("Time [min]"); plt.ylabel("Heat loss [kW]")
        plt.title("Heat loss to ambient"); plt.grid(True)
    
        # 3) Axial profiles: x=z, y=T
        z = (np.arange(Nz) + 0.5) * (H_m / Nz)
        plt.figure()
        for i in idxs:
            plt.plot(z, T_hist[i], label=f"t={time_s[i]/60:.0f} min")
        plt.xlabel("Height z [m]"); plt.ylabel("Temperature [°C]")
        plt.title("Axial temperature profiles (bottom→top)")
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"(Plots skipped: {e})")

if __name__ == "__main__":
    main()

