import numpy as np
import pyadrc
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from scipy.optimize import minimize

app = FastAPI()


class Data(BaseModel):
    SetPoint: float
    ProcessVariable: float
    ControlVariable: float
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


def pi_controller(set_point, process_variable):
    global errorSum
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

'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR MPC
'''
# System parameters
K = 0.925156  # Gain
T = 3.45  # Time constant
T0 = 0.1  # Dead time
H = 0.1  # Sampling time
# Cost function parameters
Q = 4.1  # State deviation weight
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


def mpc_controller(y, setpoint_sequence, u, horizon):

    # Define bounds for control inputs
    bounds = [(u_min, u_max)] * horizon

    # Set initial control sequence
    # u_sequence_initial = np.zeros(horizon)
    u_sequence_initial = [u] * horizon

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


'''
DECLARING GLOBAL CONSTANTS, CONTRAINTS AND VARIABLES FOR ADRC
'''
b0 = 0.26816115942028985507246376811594
delta = 0.1
order = 1
t_settle = 0.5714
k_eso = 10

adrc_statespace = pyadrc.StateSpace(order, delta, b0, t_settle, k_eso, m_lim=(0, 4), r_lim=(-1, 1))


@app.post("/cloud-controller-endpoint")
def your_endpoint(data: Data):
    set_point = data.SetPoint
    process_variable = data.ProcessVariable
    control_variable = data.ControlVariable
    controller_type = data.ControllerType

    match controller_type:
        case "PID":
            output = pi_controller(set_point, process_variable)

        case "MPC":
            horizon = 80
            setpoint_sequence = [set_point] * horizon
            # output = 1.2345
            output = mpc_controller(process_variable, setpoint_sequence, control_variable, horizon)

        case "ADRC":
            output = adrc_statespace(process_variable, control_variable, set_point)
            # print(f"SP: {set_point}, PV: {process_variable}, CV: {control_variable}, output: {output}")

        case "myADRC":
            output = 1.2345

    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
