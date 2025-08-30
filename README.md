# Sensible Stratified Heat Storage Tank Models
Lightweight, numerically robust Python models for one-dimensional vertically stratified sensible heat storage. 

The package provides four interchangeable solvers:
+ `plugflow.py` – ideal piston/plug-flow with distributed heat losses.
+ `multinode.py` – lumped multi-layer model with advective exchange + losses.
+ `fd_cn.py` – finite-difference θ-scheme (Crank–Nicolson when theta=0.5) for the 1D advection–diffusion–loss PDE with upwind advection and automatic sub-stepping.
+ `fv_cn.py` – finite-volume θ-scheme (Crank–Nicolson when theta=0.5), upwind advection, conservative flux form, south↔north naming (south=bottom, north=top).

All solvers share the same inputs and outputs, so you can swap methods without changing the calling code.

$$
\rho\,c_p\,\frac{\partial T}{\partial t}
+\rho\,c_p\,u\,\frac{\partial T}{\partial z}
=
\rho\,c_p\,D_{\mathrm{ax}}\,\frac{\partial^2 T}{\partial z^2}
- H(z)\,\big[T - T_\infty(t)\big] \, .
$$

