import numpy as np
import pyadrc
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scipy import signal
from scipy.optimize import minimize

app = FastAPI()


class Data(BaseModel):
    SetPoint: float
    ProcessVariable: float
    ControlVariable: float
    ErrorSum: float
    ControllerType: str


'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR PID
'''
LOWERLIMIT = 0
UPPERLIMIT = 4
TI = 3.45
H = 0.1
KR = 1.202936
errorSum = 0


def pi_controller(set_point, process_variable, error_sum):
    global errorSum
    errorSum = error_sum
    error = set_point - process_variable
    errorSum += error
    # output = kr * error + (kr * h / Ti) * errorSum + kr * Td * (error - errorPrev) / h #PID
    output = KR * error + (KR * H / TI) * errorSum
    if output >= UPPERLIMIT:
        output = UPPERLIMIT
        errorSum -= error
    elif output <= LOWERLIMIT:
        output = LOWERLIMIT
        errorSum -= error
    return output


'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR MPC
'''
# System parameters
K = 0.925156  # Gain
T = 3.45  # Time constant
T0 = 0.1  # Dead time
H = 0.1  # Sampling time
# Cost function parameters
Q = 100.0  # State deviation weight
R = 0.1  # Control effort weight
# Control input constraints
u_min = 0.0
u_max = 4.0


# MY MPC
y_prev_mpc = 0.0
# def system_model(y, u):
#     global y_prev_mpc

#     y = y_prev_mpc
#     delta_y = (-1 / T) * y + (K / T) * u
#     y_prev_mpc = y + delta_y * H
#     return y


def system_model(y, u):
    delta_y = (-1 / T) * y + (K / T) * u
    y += delta_y * H
    return y


def cost_function(u_sequence, y, setpoint_sequence):
    global y_prev_mpc
    cost = 0.0
    y_prev_mpc = y
    for sp, u in zip(setpoint_sequence, u_sequence):
        y_predicted = system_model(y, u)
        cost += Q * (sp - y_predicted)**2 + R * u**2
        y = y_predicted
    return cost


def mpc_controller(y, set_point, u, horizon):

    # Define bounds for control inputs
    # bounds = [(u_min, u_max)] * horizon

    # setpoint_sequence = [set_point] * horizon

    # # Set initial control sequence
    # # u_sequence_initial = np.zeros(horizon)
    # u_sequence_initial = [u] * horizon

    bounds = [(u_min, u_max)] * horizon
    setpoint_sequence = np.full(horizon, set_point)
    u_sequence_initial = np.full(horizon, u)

    # Define optimization problem
    optimization_result = minimize(
        cost_function,
        u_sequence_initial,
        args=(y, setpoint_sequence),
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 5}
    )

    # Extract optimal control sequence
    u_sequence_optimal = optimization_result.x

    # Return next optimal control input
    return u_sequence_optimal[0]


numerator = [0, 0.02643]
denominator = [1, -0.9714]


def system_model2(y, u_sequence):
    # return np.array(y + signal.lfilter(numerator, denominator, u_sequence))
    return np.concatenate((np.array([y]), np.array(y + signal.lfilter(numerator, denominator, u_sequence[0:len(u_sequence)-1]))))


# q = 10.0  # State deviation weight
# r = 0.1  # Control effort weight
q = 100.0  # State deviation weight
r = 0.1  # Control effort weight
# q = 1000.0  # State deviation weight
# r = 0.1  # Control effort weight


def cost_function2(u_sequence, y, setpoint_sequence):
    y_predicted = system_model2(y, u_sequence)
    return np.sum(q * (setpoint_sequence - y_predicted)**2 + r * u_sequence**2)


def mpc_controller2(y, set_point, u, horizon):

    # Define bounds for control inputs
    bounds = [(u_min, u_max)] * horizon
    setpoint_sequence = np.full(horizon, set_point)
    u_sequence_initial = np.full(horizon, u)

    # Define optimization problem
    optimization_result = minimize(
        cost_function2,
        u_sequence_initial,
        args=(y, setpoint_sequence),
        bounds=bounds,
        method='SLSQP',
        # options={'maxiter': 15}
        options={'maxiter': 15}
    )

    # Extract optimal control sequence
    u_sequence_optimal = optimization_result.x

    # Return next optimal control input
    return u_sequence_optimal[0]


'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR ADRC
'''
b0 = K / T
delta = H
order = 1
w_cl = 1
k_eso = 10

adrc_statespace = pyadrc.StateSpace(order, delta, b0, w_cl, k_eso, m_lim=(0, 4), r_lim=(-1, 1))


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

        self.Lc = np.array([1 - (z_eso)**2, ((1 - z_eso)**2) * (1 / Ts)]).reshape(-1, 1)
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

        # self.xhat = self.A_eso.dot(self.xhat) + self.B_eso.dot(ukm1).reshape(-1, 1) + self.Lc.dot(y)
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


b0 = K / T
delta = H
order = 1
w_cl = 1
k_eso = 10

myadrc = FirstOrderADRC(Ts=H, b0=b0, T_set=5, k_cl=4, k_eso=10, r_lim=(-1, 1), m_lim=(0, 4))


@app.post("/cloud-controller-endpoint")
def cloud_endpoint(data: Data):

    set_point = data.SetPoint
    process_variable = data.ProcessVariable
    control_variable = data.ControlVariable
    error_sum = data.ErrorSum
    controller_type = data.ControllerType

    match controller_type:

        case "PID0":
            output = pi_controller(set_point, process_variable, error_sum)

        case "MPC0":
            horizon = 50
            output = mpc_controller(process_variable, set_point, control_variable, horizon)

        case "MPC1":
            # horizon = 20
            horizon = 20
            output = mpc_controller2(process_variable, set_point, control_variable, horizon)

        case "ADRC0":
            output = adrc_statespace(process_variable, control_variable, set_point)

        case "ADRC1":
            output = myadrc(process_variable, control_variable, set_point)

    # print(f"SP: {set_point}, PV: {process_variable}, CV: {control_variable}, output: {output}")
    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
