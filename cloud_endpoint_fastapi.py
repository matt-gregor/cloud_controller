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
L = 0.1  # Dead time
Ts = 0.1  # Sampling time


# Discrete-time system model
def system_model(y, u):
    return np.exp(-L/Ts) * y + K*(1 - np.exp(-L/Ts)) * u


# Cost function parameters
Q = 1.0  # State deviation weight
R = 0.1  # Control effort weight

# MPC controller function
def mpc_controller(setpoint_sequence, horizon):
    # Control input constraints
    u_min = 0.0
    u_max = 4.0

    # Initial control sequence
    u_sequence = np.zeros(horizon)

    # Define optimization problem
    def cost_function(u_sequence):
        J = 0.0
        y = 0.0

        for i in range(horizon):
            y = system_model(y, u_sequence[i])
            J += Q * (y - setpoint_sequence[i])**2
            J += R * u_sequence[i]**2

        return J

    # Define optimization bounds
    bounds = [(u_min, u_max)] * horizon

    # Solve the optimization problem
    result = minimize(cost_function, u_sequence, bounds=bounds)

    return result.x[0]


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
            horizon = 10
            setpoint_sequence = [SP] * horizon
            output = mpc_controller(setpoint_sequence, horizon)
            # output = 1.2345
        case "ADRC":
            output = adrc_statespace(PV, CV, SP)
            print(f"SP: {SP}, PV: {PV}, CV: {CV}, output: {output}")

    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
