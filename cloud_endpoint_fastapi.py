import pyadrc
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    SetPoint: float
    ProcessVariable: float
    ControlVariable: float
    ControllerType: str


# DECLARING GLOBAL CONSTANTS AND GLOBAL VARIABLES FOR PID
LOWERLIMIT = 0
UPPERLIMIT = 4

TI = 3.45
H = 0.1
KR = 1.202936

errorSum = 0

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
            output = 1.2345
        case "ADRC":
            output = adrc_statespace(PV, CV, SP)
            print(f"SP: {SP}, PV: {PV}, CV: {CV}, output: {output}")

    result = str(output)
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
