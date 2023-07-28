import asyncio
import os
import socket
import time

import influxdb_client
import numpy as np
import pyadrc
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from pydantic import BaseModel
from scipy import signal
from scipy.optimize import minimize

app = FastAPI()

class Data(BaseModel):
    SetPoint: float
    ProcessVariable: float
    ControlVariable: float
    ErrorSum: float
    MPCHorizonLength: int
    MPCQ: float
    MPCR: float
    ControllerType: str


'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR PID
'''
LOWERLIMIT = 0.0
UPPERLIMIT = 4.0
TI = 3.45
H = 0.1
KR = 1.202936
errorSum = 0


def pi_controller(set_point, process_variable, error_sum):
    global errorSum
    if errorSum == 0:
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
DECLARING GLOBAL CONSTANTS FOR MPC
'''
# System parameters
K = 0.925156  # Gain
T = 3.45  # Time constant
T0 = 0.1  # Dead time
H = 0.1  # Sampling time
# Cost function parameters
# Q = 100.0  # State deviation weight
# R = 0.1  # Control effort weight

def system_model(y, u):
    delta_y = (-1 / T) * y + (K / T) * u
    y += delta_y * H
    return y


def cost_function(u_sequence, y, setpoint_sequence, q, r):
    cost = 0.0
    for sp, u in zip(setpoint_sequence, u_sequence):
        y_predicted = system_model(y, u)
        cost += q * (sp - y_predicted)**2 + r * u**2
        y = y_predicted
    return cost


def mpc_controller(y, set_point, u, horizon, q, r):

    # Define bounds for control inputs
    bounds = [(LOWERLIMIT, UPPERLIMIT)] * horizon

    # Set initial control sequence
    setpoint_sequence = np.full(horizon, set_point)
    u_sequence_initial = np.full(horizon, u)

    # Define optimization problem
    optimization_result = minimize(
        cost_function,
        u_sequence_initial,
        args=(y, setpoint_sequence, q, r),
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


def system_model1(y, u_sequence):
    # return np.array(y + signal.lfilter(numerator, denominator, u_sequence))
    return np.concatenate((np.array([y]), np.array(y + signal.lfilter(numerator, denominator, u_sequence[0:len(u_sequence)-1]))))


# q = 10.0  # State deviation weight
# r = 0.1  # Control effort weight
# q = 100.0  # State deviation weight
# r = 0.1  # Control effort weight
# q = 1000.0  # State deviation weight
# r = 0.1  # Control effort weight


def cost_function1(u_sequence, y, setpoint_sequence, q, r, u):
    y_predicted = system_model1(y, u_sequence)
    return np.sum(q * (setpoint_sequence - y_predicted)**2 + r * (u_sequence - np.concatenate((np.array([u]), u_sequence[0:-1])))**2)


def mpc_controller1(y, set_point, u, horizon, q, r):

    # Define bounds for control inputs
    bounds = [(LOWERLIMIT, UPPERLIMIT)] * horizon
    setpoint_sequence = np.full(horizon, set_point)
    u_sequence_initial = np.full(horizon, u)

    # Define optimization problem
    optimization_result = minimize(
        cost_function1,
        u_sequence_initial,
        args=(y, setpoint_sequence, q, r, u),
        bounds=bounds,
        method='SLSQP',
        # options={'maxiter': 15}
        options={'maxiter': 5}
    )

    # Extract optimal control sequence
    u_sequence_optimal = optimization_result.x

    # Return next optimal control input
    return u_sequence_optimal[0]


def cost_function2(u_sequence, y, setpoint_sequence, q, r, u_prev):
    cost = 0.0
    for sp, u in zip(setpoint_sequence, u_sequence):
        y_predicted = system_model(y, u)
        cost += q * (sp - y_predicted)**2 + r * (u - u_prev)**2
        y = y_predicted
        u_prev = u
    return cost


def mpc_controller2(y, set_point, u, horizon, q, r):

    # Define bounds for control inputs
    bounds = [(LOWERLIMIT, UPPERLIMIT)] * horizon

    # Set initial control sequence
    setpoint_sequence = np.full(horizon, set_point)
    u_sequence_initial = np.full(horizon, u)

    # Define optimization problem
    optimization_result = minimize(
        cost_function2,
        u_sequence_initial,
        args=(y, setpoint_sequence, q, r, u),
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 5}
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

adrc_statespace = pyadrc.StateSpace(order, delta, b0, w_cl, k_eso, m_lim=(LOWERLIMIT, UPPERLIMIT), r_lim=(-1, 1))


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

myadrc = FirstOrderADRC(Ts=H, b0=b0, T_set=5, k_cl=4, k_eso=10, r_lim=(-1, 1), m_lim=(LOWERLIMIT, UPPERLIMIT))

load_dotenv(find_dotenv())

token = os.environ.get("INFLUXDB_TOKEN")
org = "cloud_controller"
url = "http://influxdb:8086"
# url = "http://16.16.220.162:8086"

write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
bucket = "process_metrics"

write_api = write_client.write_api(write_options=SYNCHRONOUS)


async def send_data_to_db():
    while True:
        if (time.perf_counter_ns() - cpa.prev_time)/1000000000 < 2:
            vector = [influxdb_client.Point("measurement").field(name, data) for name, data in zip(cpa.names, cpa.return_values())]
            write_api.write(bucket=bucket, org=org, record=vector)
        await asyncio.sleep(1)


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

        # Previous controller type (needed for zeroing)
        self.previous_controller_type = None

        # Control variable
        self.previous_control_variable = None

        # Setpoints
        self.previous_set_point = None
        self.set_point_diff = None
        self.old_set_point = None

        # Initializing timers
        self.prev_time = time.perf_counter_ns()
        self.start_time = self.prev_time
        self.start_time_10_percent = None
        self.current_step_time = None

        # Names
        self.names = ['set_point',
                      'process_variable',
                      'control_variable',
                      'controller_type',
                      'current_error',
                      'overshoot',
                      'regulation_time',
                      'rise_time',
                      'ISE',
                      'IAE',
                      'MSE',
                      'control_cost']

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
        self.previous_set_point = self.set_point
        self.set_point = sp
        self.process_variable = y
        self.previous_control_variable = self.control_variable
        self.control_variable = u
        self.previous_controller_type = self.controller_type
        self.controller_type = ct

        if self.previous_set_point and self.previous_set_point != self.set_point:
            self.set_point_diff = self.set_point - self.previous_set_point
            self.old_set_point = self.previous_set_point
            self._zeroing()

        if self.previous_controller_type and self.previous_controller_type != self.controller_type:
            self._zeroing()

        self.current_error = self.set_point - self.process_variable
        self.error_sum += self.current_error
        self.counter += 1

        # Time in seconds
        self.current_step_time = (time.perf_counter_ns() - self.prev_time)/1000000000
        self.prev_time = time.perf_counter_ns()
        if self.old_set_point and self.set_point != 0.0:
            # Overshoot
            if self.set_point > self.old_set_point and (self.process_variable - self.set_point)/self.set_point > self.overshoot:
                self.overshoot = (self.process_variable - self.set_point)/self.set_point
            elif self.set_point < self.old_set_point and (self.process_variable - self.set_point)/self.set_point > self.overshoot:
                self.overshoot = (self.set_point - self.process_variable)/self.set_point

            # Regulation time old_sp -> 0.95*new_sp
            if self.regulation_time == 0.0:
                if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.95*self.set_point_diff:
                    self.regulation_time = (self.prev_time - self.start_time)/1000000000
                elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.95*self.set_point_diff:
                    self.regulation_time = (self.prev_time - self.start_time)/1000000000

            # Rise time 0.1*new_sp -> 0.90*new_sp TIMER START
            if self.start_time_10_percent is None:
                if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.1*self.set_point_diff:
                    self.start_time_10_percent = self.prev_time
                elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.1*self.set_point_diff:
                    self.start_time_10_percent = self.prev_time

            # Rise time 0.1*new_sp -> 0.90*new_sp
            if self.start_time_10_percent and self.rise_time == 0.0:
                if self.set_point > self.old_set_point and self.process_variable >= self.old_set_point + 0.9*self.set_point_diff:
                    self.rise_time = (self.prev_time - self.start_time_10_percent)/1000000000
                elif self.set_point < self.old_set_point and self.process_variable <= self.old_set_point + 0.9*self.set_point_diff:
                    self.rise_time = (self.prev_time - self.start_time_10_percent)/1000000000

        # Integral criteria
        self.ISE += self.current_error**2
        self.IAE += abs(self.current_error)

        # Mean Squared Error
        self.MSE = (1/self.counter) * (self.MSE + (self.current_error - (self.error_sum/self.counter))**2)

        # Control cost
        if self.previous_control_variable:
            self.control_cost += abs(self.control_variable - self.previous_control_variable)

    def return_values(self):
        return [self.set_point,
                self.process_variable,
                self.control_variable,
                self.controller_type,
                self.current_error,
                self.overshoot,
                self.regulation_time,
                self.rise_time,
                self.ISE,
                self.IAE,
                self.MSE,
                self.control_cost]


cpa = ControlPerformanceAssesment()


@app.post("/cloud-controller-endpoint")
def cloud_endpoint(data: Data):
    time1 = time.perf_counter_ns()
    set_point = data.SetPoint
    process_variable = data.ProcessVariable
    control_variable = data.ControlVariable
    error_sum = data.ErrorSum
    mpc_horizon = data.MPCHorizonLength
    mpc_q = data.MPCQ
    mpc_r = data.MPCR
    controller_type = data.ControllerType

    cpa.update_CPA_metrics(process_variable, control_variable, set_point, controller_type)
    # print(f"err: {cpa.current_error}, over: {cpa.overshoot}, reg_t: {cpa.regulation_time}, ris_t: {cpa.rise_time}, ISE: {cpa.ISE}, IAE: {cpa.IAE}, MSE: {cpa.MSE}, ctrl_c: {cpa.control_cost}")
    # send_data_to_db()

    # Ensure bumpless switching for PID
    if controller_type != "PID0":
        global errorSum
        errorSum = 0

    match controller_type:

        case "PID0":
            output = pi_controller(set_point, process_variable, error_sum)

        case "MPC0":
            # horizon = 30
            output = mpc_controller(process_variable, set_point, control_variable, mpc_horizon, mpc_q, mpc_r)

        case "MPC1":
            # horizon = 20
            output = mpc_controller1(process_variable, set_point, control_variable, mpc_horizon, mpc_q, mpc_r)

        case "MPC2":

            output = mpc_controller2(process_variable, set_point, control_variable, mpc_horizon, mpc_q, mpc_r)

        case "ADRC0":
            output = adrc_statespace(process_variable, control_variable, set_point)

        case "ADRC1":
            output = myadrc(process_variable, control_variable, set_point)

    result = output
    time2 = (time.perf_counter_ns() - time1)/1000000
    return {"result": result, "operation_time": time2, "cpa": cpa.return_values()}


@app.on_event('startup')
async def service_tasks_startup():
    # Start all the non-blocking service tasks, which run in the background
    asyncio.create_task(send_data_to_db())

if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
