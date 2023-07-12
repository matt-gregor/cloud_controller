import socket
import time

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


def system_model(y, u):
    delta_y = (-1 / T) * y + (K / T) * u
    y += delta_y * H
    return y


def cost_function(u_sequence, y, setpoint_sequence):
    cost = 0.0
    for sp, u in zip(setpoint_sequence, u_sequence):
        y_predicted = system_model(y, u)
        cost += Q * (sp - y_predicted)**2 + R * u**2
        y = y_predicted
    return cost


def mpc_controller(y, set_point, u, horizon):

    # Define bounds for control inputs
    bounds = [(u_min, u_max)] * horizon

    # Set initial control sequence
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


class ControlPerformanceAssesment():
    def __init__(self):
        # Base values of input variables for calling
        self.set_point = None
        self.process_variable = None
        self.control_variable = None
        self.controller_type = None

        # Base values for calculated metrics
        self.current_error = 0.0
        self.error_sum = 0.0
        self.overshoot = 0.0
        self.regulation_time = 0.0
        self.rise_time = 0.0
        self.ISE = 0.0
        self.IAE = 0.0
        self.MSE = 0.0
        self.control_cost = 0.0

        # Counter
        self.counter = 0

        # Control variable
        self.previous_control_variable = None

        # Setpoints
        self.prev_set_point = None
        self.set_point_diff = None
        self.old_set_point = None

        # Initializing timers
        self.prev_time = time.perf_counter_ns()
        self.start_time = time.perf_counter_ns()
        self.start_time_10_percent = None
        self.current_step_time = None

    def _zeroing(self):
        self.current_error = 0.0
        self.error_sum = 0.0
        self.overshoot = 0.0
        self.regulation_time = 0.0
        self.rise_time = 0.0
        self.ISE = 0.0
        self.IAE = 0.0
        self.MSE = 0.0
        self.control_cost = 0.0
        self.counter = 0
        self.start_time = time.perf_counter_ns()
        self.start_time_10_percent = None

    def update_CPA_metrics(self, y: float, u: float, sp: float, ct: str):
        self.prev_set_point = self.set_point
        self.set_point = sp
        self.process_variable = y
        self.previous_control_variable = self.control_variable
        self.control_variable = u
        self.controller_type = ct

        self.current_error = self.set_point - self.process_variable
        self.error_sum += self.current_error
        self.counter += 1

        if self.prev_set_point and self.prev_set_point != self.set_point:
            self.set_point_diff = self.set_point - self.prev_set_point
            self.old_set_point = self.prev_set_point
            self._zeroing()

        # Time in seconds
        self.current_step_time = (time.perf_counter_ns() - self.prev_time)/1000000000
        self.prev_time = time.perf_counter_ns()

        # Overshoot
        if self.set_point > self.old_set_point:
            self.overshoot = (self.process_variable - self.set_point)/self.set_point
        elif self.set_point < self.old_set_point:
            self.overshoot = (self.set_point - self.process_variable)/self.set_point

        # Regulation time old_sp -> 0.95*new_sp
        if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.95*self.set_point_diff:
            self.regulation_time = (time.perf_counter_ns() - self.start_time)/1000000000
        elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.95*self.set_point_diff:
            self.regulation_time = (time.perf_counter_ns() - self.start_time)/1000000000

        # Rise time 0.1*new_sp -> 0.90*new_sp TIMER START
        if self.start_time_10_percent is None:
            if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.1*self.set_point_diff:
                self.start_time_10_percent = time.perf_counter_ns()
            elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.1*self.set_point_diff:
                self.start_time_10_percent = time.perf_counter_ns()

        # Rise time 0.1*new_sp -> 0.90*new_sp
        if self.start_time_10_percent:
            if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.9*self.set_point_diff:
                self.rise_time = (time.perf_counter_ns() - self.start_time_10_percent)/1000000000
            elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.9*self.set_point_diff:
                self.rise_time = (time.perf_counter_ns() - self.start_time_10_percent)/1000000000

        # Integral criteria
        self.ISE += (self.current_error**2) * self.current_step_time
        self.IAE += abs(self.current_error * self.current_step_time)

        # Mean Squared Error
        self.MSE = (1/self.counter) * (self.MSE + (self.current_error - (self.error_sum/self.counter)**2))

        # Control cost
        if self.previous_control_variable:
            self.control_cost += abs(self.control_variable - self.previous_control_variable)


cpa = ControlPerformanceAssesment()


@app.post("/cloud-controller-endpoint")
def cloud_endpoint(data: Data):

    set_point = data.SetPoint
    process_variable = data.ProcessVariable
    control_variable = data.ControlVariable
    error_sum = data.ErrorSum
    controller_type = data.ControllerType

    cpa.update_CPA_metrics(process_variable, control_variable, set_point, controller_type)

    match controller_type:

        case "PID0":
            output = pi_controller(set_point, process_variable, error_sum)

        case "MPC0":
            horizon = 30
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

    # UDP_IP = "mynodered"
    # UDP_PORT = 5005

    # message_to_node_red = f"{set_point} {process_variable} {control_variable} {controller_type}".encode('utf-8')

    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.sendto(message_to_node_red, (UDP_IP, UDP_PORT))

    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
