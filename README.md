# Sensible Stratified Heat Storage Tank Models
Lightweight, numerically robust Python models for one-dimensional vertically stratified sensible heat storage. 

The package provides four interchangeable solvers:
+ `plugflow.py` – ideal piston/plug-flow with distributed heat losses.
+ `multinode.py` – lumped multi-layer model with advective exchange + losses.
+ `fd_cn.py` – finite-difference θ-scheme (Crank–Nicolson when theta=0.5) for the 1D advection–diffusion–loss PDE with upwind advection and automatic sub-stepping.
+ `fv_cn.py` – finite-volume θ-scheme (Crank–Nicolson when theta=0.5), upwind advection, conservative flux form, south↔north naming (south=bottom, north=top).

All solvers share the same inputs and outputs, so you can swap methods without changing the calling code.

## Problem
We simulate the axial temperature field 𝑇(𝑧,𝑡) in a vertical tank with single inlet/outlet, axial advection, axial dispersion/diffusion, and heat loss to ambient together with an inflow boundary at the inlet (imposed 𝑇in) and convective outflow at the outlet 𝐻(𝑧) collecting side/top/bottom loss conductances:

**Governing PDE**\
$$\rho c_p \frac{\partial T}{\partial t} + \rho c_p u \frac{\partial T}{\partial z} = \rho c_p D_{\mathrm{ax}} \frac{\partial^2 T}{\partial z^2} - H(z),[T - T_\infty(t)]$$

## Repository Layout
```
.
├── examplescript.py      # simple driver; edit parameters at the top
├── plugflow.py           # run_plug_flow(...)
├── multinode.py          # run_multi_node(...)
├── fd_cn.py              # run_fd_cn(...)  (FD θ-scheme)
└── fv_cn.py              # run_fv_cn(...)  (FV θ-scheme, south/north terms)

```
