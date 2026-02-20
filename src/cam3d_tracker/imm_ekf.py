from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .math_utils import angle_diff, wrap_angle

STATE_DIM = 9
MEAS_DIM = 7


@dataclass
class IMMState:
    x_models: list[np.ndarray]
    p_models: list[np.ndarray]
    mu: np.ndarray


class IMMEKF:
    def __init__(self, x0: np.ndarray, p0: np.ndarray, mode_prob_init: np.ndarray, transition: np.ndarray):
        self.transition = transition
        self.state = IMMState(
            x_models=[x0.copy(), x0.copy()],
            p_models=[p0.copy(), p0.copy()],
            mu=mode_prob_init.astype(float).copy(),
        )
        self.x = x0.copy()
        self.p = p0.copy()
        self._fuse()

    @staticmethod
    def _f_cv(x: np.ndarray, dt: float) -> np.ndarray:
        xn = x.copy()
        px, py, pz, v, yaw, yaw_rate, l, w, h = x
        xn[0] = px + v * dt * math.cos(yaw)
        xn[1] = py + v * dt * math.sin(yaw)
        xn[2] = pz
        xn[3] = v
        xn[4] = wrap_angle(yaw)
        xn[5] = 0.95 * yaw_rate
        xn[6] = max(0.05, l)
        xn[7] = max(0.05, w)
        xn[8] = max(0.05, h)
        return xn

    @staticmethod
    def _f_ctrv(x: np.ndarray, dt: float) -> np.ndarray:
        xn = x.copy()
        px, py, pz, v, yaw, yaw_rate, l, w, h = x
        if abs(yaw_rate) > 1e-4:
            xn[0] = px + (v / yaw_rate) * (math.sin(yaw + yaw_rate * dt) - math.sin(yaw))
            xn[1] = py - (v / yaw_rate) * (math.cos(yaw + yaw_rate * dt) - math.cos(yaw))
        else:
            xn[0] = px + v * dt * math.cos(yaw)
            xn[1] = py + v * dt * math.sin(yaw)
        xn[2] = pz
        xn[3] = v
        xn[4] = wrap_angle(yaw + yaw_rate * dt)
        xn[5] = yaw_rate
        xn[6] = max(0.05, l)
        xn[7] = max(0.05, w)
        xn[8] = max(0.05, h)
        return xn

    @staticmethod
    def _h(x: np.ndarray) -> np.ndarray:
        z = np.array([x[0], x[1], x[2], x[4], x[6], x[7], x[8]], dtype=float)
        z[3] = wrap_angle(z[3])
        return z

    @staticmethod
    def _jacobian_numeric(func, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        y0 = func(x)
        j = np.zeros((y0.size, x.size), dtype=float)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            yp = func(xp)
            j[:, i] = (yp - y0) / eps
        return j

    def _mix(self) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        mu_prev = self.state.mu
        c_j = self.transition.T @ mu_prev
        c_j = np.maximum(c_j, 1e-12)

        mixed_x: list[np.ndarray] = []
        mixed_p: list[np.ndarray] = []

        for j in range(2):
            mu_ij = (self.transition[:, j] * mu_prev) / c_j[j]
            xj = sum(mu_ij[i] * self.state.x_models[i] for i in range(2))
            pj = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
            for i in range(2):
                dx = self.state.x_models[i] - xj
                dx[4] = angle_diff(self.state.x_models[i][4], xj[4])
                pj += mu_ij[i] * (self.state.p_models[i] + np.outer(dx, dx))
            mixed_x.append(xj)
            mixed_p.append(pj)

        return mixed_x, mixed_p, c_j

    def predict(self, dt: float, q_cv: np.ndarray, q_ctrv: np.ndarray) -> None:
        mixed_x, mixed_p, c_j = self._mix()

        funcs = [self._f_cv, self._f_ctrv]
        qs = [q_cv, q_ctrv]

        x_pred: list[np.ndarray] = []
        p_pred: list[np.ndarray] = []
        for j in range(2):
            xj = funcs[j](mixed_x[j], dt)
            fj = self._jacobian_numeric(lambda xx: funcs[j](xx, dt), mixed_x[j])
            pj = fj @ mixed_p[j] @ fj.T + qs[j]
            x_pred.append(xj)
            p_pred.append(pj)

        self.state.x_models = x_pred
        self.state.p_models = p_pred
        self.state.mu = c_j / np.sum(c_j)
        self._fuse()

    def update(self, z: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        likelihoods = np.zeros(2, dtype=float)
        x_upd: list[np.ndarray] = []
        p_upd: list[np.ndarray] = []

        for j in range(2):
            xj = self.state.x_models[j]
            pj = self.state.p_models[j]

            hj = self._h(xj)
            innov = z - hj
            innov[3] = angle_diff(z[3], hj[3])

            h_jac = self._jacobian_numeric(self._h, xj)
            s = h_jac @ pj @ h_jac.T + r
            s = 0.5 * (s + s.T)
            s_inv = np.linalg.inv(s)
            k = pj @ h_jac.T @ s_inv

            xu = xj + k @ innov
            xu[4] = wrap_angle(xu[4])
            iu = np.eye(STATE_DIM)
            pu = (iu - k @ h_jac) @ pj
            pu = 0.5 * (pu + pu.T)

            det_s = max(np.linalg.det(s), 1e-12)
            mahal = float(innov.T @ s_inv @ innov)
            norm = math.sqrt(((2 * math.pi) ** MEAS_DIM) * det_s)
            likelihoods[j] = math.exp(-0.5 * mahal) / norm

            x_upd.append(xu)
            p_upd.append(pu)

        mu = self.state.mu * np.maximum(likelihoods, 1e-20)
        mu = mu / np.sum(mu)

        self.state.x_models = x_upd
        self.state.p_models = p_upd
        self.state.mu = mu
        self._fuse()

        z_hat = self._h(self.x)
        h_jac = self._jacobian_numeric(self._h, self.x)
        s_fused = h_jac @ self.p @ h_jac.T + r
        return z_hat, s_fused

    def innovation_mahalanobis(self, z: np.ndarray, r: np.ndarray) -> float:
        z_hat = self._h(self.x)
        innov = z - z_hat
        innov[3] = angle_diff(z[3], z_hat[3])
        h_jac = self._jacobian_numeric(self._h, self.x)
        s = h_jac @ self.p @ h_jac.T + r
        s = 0.5 * (s + s.T)
        s_inv = np.linalg.inv(s)
        return float(innov.T @ s_inv @ innov)

    def _fuse(self) -> None:
        mu = self.state.mu
        xf = sum(mu[i] * self.state.x_models[i] for i in range(2))
        pf = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        for i in range(2):
            dx = self.state.x_models[i] - xf
            dx[4] = angle_diff(self.state.x_models[i][4], xf[4])
            pf += mu[i] * (self.state.p_models[i] + np.outer(dx, dx))
        self.x = xf
        self.x[4] = wrap_angle(self.x[4])
        self.p = 0.5 * (pf + pf.T)
