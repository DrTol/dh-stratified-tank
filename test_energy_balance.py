"""Synthetic regressions for insulated diffusion and idle heat accounting.

Run: python -m unittest discover -v
Only NumPy and the standard library are required. No field validation is implied.
"""
import unittest
import numpy as np
import fd_cn
import fv_cn


class EnergyBalanceTests(unittest.TestCase):
    def run_idle(self, module, initial, theta, losses, diffusion=1e-4):
        n = len(initial)
        params = module.TankParams(H=1., A=1., Nz=n, rho=1000., cp=4180.,
                                   UA_side_per_m=losses, UA_top=losses / 2,
                                   UA_bot=losses / 3)
        times = np.array([0., 600., 1800., 3600.])
        run = getattr(module, 'run_' + ('fd' if module is fd_cn else 'fv') + '_cn')
        result = run(times, np.zeros(3), np.full(3, 80.),
                     np.array([20., 30., 10.]), np.array(initial), params,
                     theta=theta, D_ax=diffusion, fo_cap=0.4)
        return times, params, result

    def test_insulated_uniform_temperature(self):
        for module in (fd_cn, fv_cn):
            for n in (1, 2, 7):
                for theta in (0., 0.5, 0.55, 1.):
                    with self.subTest(module=module.__name__, n=n, theta=theta):
                        _, _, result = self.run_idle(module, [80.] * n, theta, 0.)
                        np.testing.assert_allclose(result['T_hist'], 80., atol=2e-10, rtol=0.)
                        np.testing.assert_array_equal(result['Q_loss_W'], 0.)

    def test_insulated_nonuniform_energy(self):
        for module in (fd_cn, fv_cn):
            for theta in (0., 0.5, 0.55, 1.):
                with self.subTest(module=module.__name__, theta=theta):
                    _, _, result = self.run_idle(module, [20., 40., 80., 60.], theta, 0.)
                    np.testing.assert_allclose(result['T_hist'].sum(axis=1), 200., atol=2e-10, rtol=0.)
                    self.assertGreater(np.ptp(result['T_hist'][0]), np.ptp(result['T_hist'][-1]))

    def test_idle_heat_ledger(self):
        # Delta stored energy + reported outward heat must vanish on EACH
        # reporting interval, including internally substepped intervals.
        for module in (fd_cn, fv_cn):
            for n in (1, 7):
                for theta in (0., 0.5, 0.55, 1.):
                    for initial in (5., 80.):
                        with self.subTest(module=module.__name__, n=n, theta=theta, initial=initial):
                            times, params, result = self.run_idle(module, [initial] * n, theta, 500.)
                            capacity = params.rho * params.A * params.H / n * params.cp
                            delta_e = capacity * np.diff(result['T_hist'].sum(axis=1))
                            loss_e = result['Q_loss_W'] * np.diff(times)
                            np.testing.assert_allclose(delta_e + loss_e, 0., atol=2e-6, rtol=0.)


if __name__ == '__main__':
    unittest.main()
