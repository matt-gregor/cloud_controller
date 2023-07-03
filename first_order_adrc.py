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
                 w_cl: float,
                 k_eso: float,
                 r_lim: tuple = (None, None),
                 m_lim: tuple = (None, None),
                 half_gain: tuple = (False, False)):

        # Discretised matrices for first order ADRC
        self.Ad = np.vstack(([1, Ts], [0, 1]))
        self.Bd = np.vstack((b0 * Ts, 0))
        self.Cd = np.hstack((1, 0)).reshape(1, -1)
        self.Dd = 0

        # Controller parameters for closed-loop dynamics
        t_settle = 4 / w_cl
        sCL = -4 / t_settle
        self.Kp = -2 * sCL

        # Observer dynamics
        sESO = k_eso * sCL
        zESO = np.exp(sESO * Ts)

        # Observer gains resulting in common-location observer poles
        self.L = np.array([1 - (zESO)**2,
                            (1 / Ts) * (1 - zESO)**2]).reshape(-1, 1)

        # Controller gains
        self.w = np.array([self.Kp / self.b0,
                            1 / self.b0]).reshape(-1, 1)

        self.xhat = np.zeros((2, 1), dtype=np.float64)

        self.ukm1 = np.zeros((1, 1), dtype=np.float64)

        self.m_lim = m_lim
        self.r_lim = r_lim

        if half_gain[0] is True:
            self.w = self.w / 2
        if half_gain[1] is True:
            self.L = self.L / 2

        self._linear_extended_state_observer()

    def _linear_extended_state_observer(self):
        """Internal function implementing the one-step update
        equation for the linear extended state observer
        """

        self.oA = self.Ad - self.L @ self.Cd @ self.Ad
        self.oB = self.Bd - self.L @ self.Cd @ self.Bd
        self.oC = self.Cd

    def _update_eso(self, y: float, ukm1: float):
        """Update the linear extended state observer

        Parameters
        ----------
        y : float
            Current measurement y[k]
        ukm1 : float
            Previous control signal u[k-1]
        """

        self.xhat = self.oA.dot(self.xhat) + self.oB.dot(
            ukm1).reshape(-1, 1) + self.L.dot(y)

    def _limiter(self, u_control: float) -> float:
        """Implements rate and magnitude limiter

        Parameters
        ----------
        u_control : float
            control signal to be limited

        Returns
        -------
        float
            float: rate and magnitude limited control signal
        """

        # Limiting the rate of u (delta_u)
        delta_u = saturation((self.r_lim[0], self.r_lim[1]),
                             u_control - self.ukm1)

        # Limiting the magnitude of u
        self.ukm1 = saturation((self.m_lim[0],
                                self.m_lim[1]),
                               delta_u + self.ukm1)

        return self.ukm1

    @property
    def limiter(self) -> tuple:
        """Returns the value of both limiters of the controller

        Returns
        -------
        tuple of tuples
            Returns (magnitude_limits, rate_limits)
        """

        return self.m_lim, self.r_lim

    @limiter.setter
    def limiter(self, lim_tuple: tuple) -> None:
        """Setter for magnitude and rate limiter

        Parameters
        ----------
            lim_tuple : tuple of tuples
                New magnitude limits
        """

        assert len(lim_tuple) == 2
        assert len(lim_tuple[0]) == 2 and len(lim_tuple[1]) == 2
        # assert lim[0] < lim[1]
        self.m_lim = lim_tuple[0]
        self.r_lim = lim_tuple[1]

    def __call__(self, y: float, u: float, r: float):

        self._update_eso(y, u)

        #control law
        u = (self.Kp / self.b0) * r - self.w.T @ self.xhat
        u = self._limiter(u)

        return float(u)

