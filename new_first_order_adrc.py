import numpy as np


def saturation(_limits: tuple, _val: float) -> float:
    lo, hi = _limits

    if _val is None:
        return None
    elif hi is not None and _val > hi:
        return hi
    elif lo is not None and _val < lo:
        return lo

    return _val


class FirstOrderADRC():
    def __init__(self,
                 Ts: float,
                 b0: float,
                 T_set: float,
                 k_cl: float,
                 k_eso: float,
                 r_lim: tuple = (None, None),
                 m_lim: tuple = (None, None)):

        self.b0 = b0

        # Discretised matrices for first order ADRC
        self.A_d = np.vstack(([1, Ts], [0, 1]))
        self.B_d = np.vstack((b0 * Ts, 0))
        self.C_d = np.hstack((1, 0)).reshape(1, -1)
        self.D_d = 0

        # Parametrization
        w_cl = k_cl / T_set
        s_cl = - 4 / T_set
        self.Kp = - s_cl
        s_eso = k_eso * s_cl
        z_eso = np.exp(s_eso * Ts)

        self.Lc = np.array([1 - (z_eso)**2, (1 - (z_eso)**2) * (1 / Ts)]).reshape(-1, 1)
        self.w = np.array([self.Kp / self.b0, 1 / self.b0]).reshape(-1, 1)

        self.xhat = np.zeros((2, 1), dtype=np.float64)
        self.ukm1 = np.zeros((1, 1), dtype=np.float64)

        self.m_lim = m_lim
        self.r_lim = r_lim

        # Current observer
        self.A_eso = self.A_d - self.Lc @ self.C_d @ self.A_d
        self.B_eso = self.B_d - self.Lc @ self.C_d @ self.B_d
        self.C_eso = self.C_d

    def _update_eso(self, y: float, ukm1: float):

        self.xhat = self.A_eso.dot(self.xhat) + self.B_eso.dot(ukm1).reshape(-1, 1) + self.Lc.dot(y)

    def _limiter(self, u_control: float) -> float:
        # Limiting the rate of u (delta_u)
        delta_u = saturation((self.r_lim[0], self.r_lim[1]),
                             u_control - self.ukm1)

        # Limiting the magnitude of u
        self.ukm1 = saturation((self.m_lim[0],
                                self.m_lim[1]),
                               delta_u + self.ukm1)

        return self.ukm1

    def __call__(self, y: float, u: float, r: float):

        self._update_eso(y, u)

        # Control law
        u = (self.Kp / self.b0) * r - self.w.T @ self.xhat
        u = self._limiter(u)

        return float(u)
