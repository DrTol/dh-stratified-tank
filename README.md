# Sensible Stratified Heat Storage Tank Models
Lightweight, numerically robust Python models for one-dimensional vertically stratified sensible heat storage. 

The package provides four interchangeable solvers:
+ `plugflow.py` – ideal piston/plug-flow with distributed heat losses.
+ `multinode.py` – lumped multi-layer model with advective exchange + losses.
+ `fd_cn.py` – finite-difference θ-scheme (Crank–Nicolson when theta=0.5) for the 1D advection–diffusion–loss PDE with upwind advection and automatic sub-stepping.
+ `fv_cn.py` – finite-volume θ-scheme (Crank–Nicolson when theta=0.5), upwind advection, conservative flux form, south↔north naming (south=bottom, north=top).

All solvers share the same inputs and outputs, so you can swap methods without changing the calling code.

## Table of Contents
- [Problem](README.md#Problem)
- [Governing PDE](README.md#Governing-PDE)
- [Repository Layout](README.md#Repository-Layout)
- [Methods](README.md#Methods)
- [Stability Conditions](README.md#Stability-Conditions)
- [License](README.md#License)
- [Acknowledgements](README.md#Acknowledgements)

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
**Dependencies**: `numpy` (required) while `matplotlib` is optional for plotting in the example script.

## Methods
### Plug-flow (`plugflow.py`)
Idealized, no axial mixing; fastest. Upper bound on stratification. Good for scenario scans or sanity bounds on return temperature.

### Multi-node (`multinode.py`)
Discretizes tank into layers with simple advective exchange and per-layer losses. Excellent accuracy/complexity trade-off for DH studies.

### FD θ-scheme (`fd_cn.py`)
PDE on a cell-centered grid, upwind advection, θ-time stepping (explicit/implicit/CN). Includes automatic sub-stepping to satisfy CFL/Fourier/sink caps and still report on the user’s time grid.

### FV θ-scheme (`fv_cn.py`)
Control-volume, conservative flux form (south/north faces), upwind advection, θ-time stepping with the same sub-stepping safeguards. Use when strict energy conservation and clear boundary handling matter.

## Stability Conditions
### Courant number (CFL):
$$\mathrm{Co}=|u|,\Delta t/\Delta z\le 1$$

### Fourier number (diffusion cap for accuracy): 
$$\mathrm{Fo}=D_{\mathrm{ax}},\Delta t/\Delta z^{2}\lesssim \text{fo_cap}$$

### Explicit sink cap (only if } \theta<1 \text{):
$$\Delta t\le \dfrac{1}{(1-\theta),S_{\max}},\quad S_{\max}=\max_i \dfrac{H_i}{C_{\text{cell}}}$$

### Grid Péclet number (diagnostic):
$$\mathrm{Pe}h=\dfrac{|u|,\Delta z}{2D{\mathrm{ax}}}$$

## License
Open-source under MIT License. Please acknowledge authorship if you use or modify.

## Acknowledgements
Above all, I give thanks to **Allah, The Creator (C.C.)**, and honor His name **Al-‘Alīm (The All-Knowing)**.

This repository is lovingly dedicated to my parents who have passed away, in remembrance of their guidance and support.

I would also like to thank **ChatGPT (by OpenAI)** for providing valuable support in updating and improving the Python implementation.



















