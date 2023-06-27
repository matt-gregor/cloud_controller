import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    SetPoint: float
    ProcessVariable: float
    ControlVariable: float


test = 15


@app.post("/your-endpoint")
def your_endpoint(data: Data):
    result = f"data received: SP = {data.SetPoint}, PV = {data.ProcessVariable}, CV = {data.ControlVariable}, test = {test}"
    return {"result": result}


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
