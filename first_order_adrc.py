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


class StateSpace():

    """Discrete linear time-invariant state space implementation\
    of ADRC

    Parameters
    ----------
    order : int
        first- or second-order ADRC
    delta : float
        sampling time in seconds
    b0 : float
        gain parameter b0
    w_cl : float
        desired closed-loop bandwidth [rad/s], 4 / w_cl and 6 / w_cl is the
        corresponding settling time in seconds for first- and second-order ADRC
        respectively
    k_eso : float
        relational observer bandwidth
    eso_init : tuple, optional
        initial state for the extended state observer, by default False
    r_lim : tuple, optional
        rate limits for the control signal, by default (None, None)
    m_lim : tuple, optional
        magnitude limits for the control signal, by default (None, None)
    half_gain : tuple, optional
        half gain tuning for controller/observer gains,\
        by default (False, False)

    References
    ----------
    .. [1] G. Herbst, "Practical active disturbance rejection control:
        Bumpless transfer, rate limitation, and incremental algorithm",
        https://arxiv.org/abs/1908.04610

    .. [2] G. Herbst, "Half-Gain Tuning for Active Disturbance Rejection
        Control", https://arxiv.org/abs/2003.03986
    """

    def __init__(self,
                 delta: float,
                 b0: float,
                 w_cl: float,
                 k_eso: float,
                 eso_init: tuple = False,
                 r_lim: tuple = (None, None),
                 m_lim: tuple = (None, None),
                 half_gain: tuple = (False, False)):

        order = 1
        self.b0 = b0
        nx = order + 1
        self.delta = delta

        if order == 1:
            self.Ad = np.vstack(([1, delta], [0, 1]))
            self.Bd = np.vstack((b0 * delta, 0))
            self.Cd = np.hstack((1, 0)).reshape(1, -1)
            self.Dd = 0

            # Controller parameters for closed-loop dynamics
            t_settle = 4 / w_cl
            sCL = -4 / t_settle
            self.Kp = -2 * sCL

            # Observer dynamics
            sESO = k_eso * sCL
            zESO = np.exp(sESO * delta)

            # Observer gains resulting in common-location observer poles
            self.L = np.array([1 - (zESO)**2,
                               (1 / delta) * (1 - zESO)**2]).reshape(-1, 1)

            # Controller gains
            self.w = np.array([self.Kp / self.b0,
                               1 / self.b0]).reshape(-1, 1)

        self.xhat = np.zeros((nx, 1), dtype=np.float64)

        self.ukm1 = np.zeros((1, 1), dtype=np.float64)

        self.m_lim = m_lim
        self.r_lim = r_lim

        if half_gain[0] is True:
            self.w = self.w / 2
        if half_gain[1] is True:
            self.L = self.L / 2

        if eso_init is not False:
            assert len(eso_init) == nx,\
                'Length of initial state vector of LESO not compatible\
                    with order'
            self.xhat = np.fromiter(eso_init, np.float64).reshape(-1, 1)

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

        u = (self.Kp / self.b0) * r - self.w.T @ self.xhat
        u = self._limiter(u)

        return float(u)

