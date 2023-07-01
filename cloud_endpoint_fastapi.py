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


##### DECLARING GLOBAL CONSTANTS AND GLOBAL VARIABLES FOR PID
LOWERLIMIT = 0
UPPERLIMIT = 4

TI = 3.45
H = 0.1
KR = 1.202936

errorSum = 0


##### MPC

# System parameters
K = 0.925156  # Gain
T = 3.45  # Time constant
T0 = 0.1  # Dead time
H = 0.1  # Sampling time
# Cost function parameters
Q = 2.1  # State deviation weight
R = 0.1  # Control effort weight
# Control input constraints
u_min = 0.0
u_max = 4.0


# MY MPC
y_prev_mpc = 0.0
def system_model(y, u):
    global y_prev_mpc

    y = y_prev_mpc
    delta_y = (-1 / T) * y + (K / T) * u
    y_prev_mpc = y + delta_y * H
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
        method='SLSQP'
    )

    # Extract optimal control sequence
    u_sequence_optimal = optimization_result.x

    # Return next optimal control input
    return u_sequence_optimal[0]


######### ADRC
b0 = 0.26816115942028985507246376811594
delta = 0.1
order = 1
t_settle = 0.5714
k_eso = 10

adrc_statespace = pyadrc.StateSpace(order, delta, b0, t_settle, k_eso, m_lim=(0, 4), r_lim=(-1, 1))


@app.post("/your-endpoint")
def your_endpoint(data: Data):
    global errorSum
    SP = data.SetPoint
    PV = data.ProcessVariable
    CV = data.ControlVariable
    ControllerType = data.ControllerType

    match ControllerType:
        case "PID":
            error = SP - PV
            errorSum += error
            # output = kr * error + (kr * h / Ti) * errorSum + kr * Td * (error - errorPrev) / h #PID
            output = KR * error + (KR * H / TI) * errorSum
            if output >= UPPERLIMIT:
                output = UPPERLIMIT
                errorSum -= error
            elif output <= LOWERLIMIT:
                output = LOWERLIMIT
                errorSum -= error

        case "MPC":
            horizon = 30
            setpoint_sequence = [SP] * horizon
            # output = 1.2345
            output = mpc_controller(PV, setpoint_sequence, CV, horizon)

        case "ADRC":
            output = adrc_statespace(PV, CV, SP)
            print(f"SP: {SP}, PV: {PV}, CV: {CV}, output: {output}")

    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
