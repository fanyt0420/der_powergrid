"""Damped Newton, consistent equilibrium, implicit Euler and DAE modes."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve
from src.dynamics.model import pack


class NumericalFailure(RuntimeError):
    def __init__(self, stage, residual, iterations, condition=None):
        super().__init__(f"{stage}: residual={residual:.3g}, iterations={iterations}")
        self.stage, self.residual, self.iterations, self.condition = stage, float(residual), iterations, condition


def derivative(function, z):
    # Only nonlinear terms are differenced. The stiff network block is exact.
    result = np.empty((len(function(z)), len(z)))
    for k in range(len(z)):
        h = 2e-6 * max(1.0, abs(z[k]))
        plus, minus = z.copy(), z.copy()
        plus[k] += h
        minus[k] -= h
        result[:, k] = (function(plus) - function(minus)) / (2 * h)
    return result


def newton(function, jacobian, initial, tolerance=1e-8, max_iterations=20, stage="newton"):
    # Keep iterates in extended precision; Jacobian/linear solve remain float64.
    # This resolves sub-microvolt drops across the 1e-7 ohm closed switch.
    z = initial.astype(np.longdouble)
    condition = None
    for iteration in range(max_iterations + 1):
        r = function(z)
        norm = float(np.max(np.abs(r)))
        if np.isfinite(norm) and norm <= tolerance:
            return z, {"residual": norm, "iterations": iteration}
        if iteration == max_iterations or not np.all(np.isfinite(r)):
            break
        jac = jacobian(z)
        try:
            step = solve(jac, -r, check_finite=False)
        except np.linalg.LinAlgError:
            condition = float(np.linalg.cond(jac))
            break
        accepted = False
        for power in range(12):
            trial = z + step * 0.5**power
            try:
                trial_norm = np.max(np.abs(function(trial)))
            except ValueError:
                continue
            if trial_norm < norm or trial_norm <= tolerance:
                z = trial
                accepted = True
                break
        if not accepted:
            condition = float(np.linalg.cond(jac))
            break
    raise NumericalFailure(stage, norm, iteration, condition)


def equilibrium(model):
    x0, y0 = model.initial_guess()
    ny = model.ny
    gfm = [d for d in model.dynamic if d["model"] != "gfl"]
    def decode(z):
        x = x0.copy()
        for k, d in enumerate(gfm):
            x[d["slice"].start] = z[ny + k]
            x[d["slice"].start + (2 if d["model"] == "inertia" else 1)] = z[ny + len(gfm) + k]
        model.initialize_gfl(x, z[:ny])
        return x, z[:ny]
    def nonlinear(z):
        x, y = decode(z)
        _, injection, records, _, _ = model.evaluate(x, y)
        records = {r["device_id"]: r for r in records}
        powers = [complex(records[d["device_id"]]["p_kw"], records[d["device_id"]]["q_kvar"]) * 1000 / d["s_ref_va"] - complex(d["p_star"], d["q_star"]) for d in gfm]
        return np.r_[pack(-injection[model.passive.free] / model.passive.ibase[model.passive.free]), np.real(powers), np.imag(powers)]
    def residual(z):
        x, y = decode(z)
        r = nonlinear(z)
        r[:ny] = model.kcl(x, y)
        return r
    def jacobian(z):
        jac = derivative(nonlinear, z)
        jac[:ny, :ny] += model.g_linear
        return jac
    angles = [x0[d["slice"].start] for d in gfm]
    amplitudes = [x0[d["slice"].start + (2 if d["model"] == "inertia" else 1)] for d in gfm]
    z, info = newton(residual, jacobian, np.r_[y0, angles, amplitudes], model.config["tolerance"], model.config["max_newton_iterations"], "initialization")
    x, y = decode(z)
    for d in gfm:
        d["e_star"] = float(x[d["slice"].start + (2 if d["model"] == "inertia" else 1)])
    records = model.evaluate(x, y)[2]
    violations = model.domain_violations(records)
    info.update(status="infeasible_or_out_of_domain" if violations else "converged", violations=violations,
                equations=len(z), dynamic_derivative_max=float(max(abs(model.evaluate(x, y)[0]))),
                kcl_max=float(max(abs(model.kcl(x, y)))))
    return x, y, info


def step(model, x_old, y_old, dt, **kwargs):
    nx = model.nx
    def nonlinear(z):
        f, injection, _, _, _ = model.evaluate(z[:nx], z[nx:], **kwargs)
        return np.r_[-dt * f, pack(-injection[model.passive.free] / model.passive.ibase[model.passive.free])]
    def residual(z):
        r = nonlinear(z)
        r[:nx] += z[:nx] - x_old
        r[nx:] = model.kcl(z[:nx], z[nx:], **kwargs)
        return r
    def fresh_jacobian(z):
        jac = derivative(nonlinear, z)
        jac[:nx, :nx] += np.eye(nx)
        jac[nx:, nx:] += model.g_linear
        return jac
    # Modified Newton: reuse a tangent for nearby steps, always testing the
    # original nonlinear residual. Refresh on events/dt changes or stagnation.
    key = (round(dt, 14), tuple(sorted(kwargs.get("scales", {}).items())), tuple(sorted(kwargs.get("tripped", ()))))
    cached = getattr(model, "_step_tangent", None)
    initial = np.r_[x_old, y_old]
    if cached is None or cached[0] != key or cached[2] >= 25:
        cached = (key, fresh_jacobian(initial), 0)
    model._step_tangent = (key, cached[1], cached[2] + 1)
    try:
        z, info = newton(residual, lambda z: cached[1], initial, model.config["tolerance"], model.config["max_newton_iterations"], "step_newton")
    except NumericalFailure:
        model._step_tangent = None
        z, info = newton(residual, fresh_jacobian, initial, model.config["tolerance"], model.config["max_newton_iterations"], "step_newton")
    return z[:nx], z[nx:], info


def algebraic_solve(model, x, y, **kwargs):
    def nonlinear(y):
        return pack(-model.evaluate(x, y, **kwargs)[1][model.passive.free] / model.passive.ibase[model.passive.free])
    return newton(lambda y: model.kcl(x, y, **kwargs), lambda y: derivative(nonlinear, y) + model.g_linear,
                  y, model.config["tolerance"], model.config["max_newton_iterations"], "event_algebraic")


def modes(model, x, y):
    nx = model.nx
    def nonlinear(z):
        f, injection, _, _, _ = model.evaluate(z[:nx], z[nx:])
        return np.r_[f, pack(-injection[model.passive.free] / model.passive.ibase[model.passive.free])]
    jac = derivative(nonlinear, np.r_[x, y])
    jac[nx:, nx:] += model.g_linear
    fx, fy, gx, gy = jac[:nx, :nx], jac[:nx, nx:], jac[nx:, :nx], jac[nx:, nx:]
    reduced = fx - fy @ solve(gy, gx)
    values = np.linalg.eigvals(reduced)
    return {"gy_condition_number": float(np.linalg.cond(gy)), "spectral_abscissa_per_s": float(max(values.real)),
            "stable": bool(np.all(values.real < 0)),
            "modes": [{"real_per_s": float(v.real), "imag_per_s": float(v.imag), "frequency_hz": float(abs(v.imag) / (2 * np.pi)), "damping_ratio": float(-v.real / abs(v)) if abs(v) else None} for v in sorted(values, key=lambda v: v.real, reverse=True)]}
