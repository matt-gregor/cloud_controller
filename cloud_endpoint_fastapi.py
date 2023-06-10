import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Data(BaseModel):
    value: int


@app.post("/your-endpoint")
def your_endpoint(data: Data):
    value = data.value

    # Perform actions based on the numerical value
    if value > 10:
        result = "Value is greater than 10"
    else:
        result = "Value is less than or equal to 10"

    return {"result": result}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8080)
