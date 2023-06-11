# import signal
# import sys

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
        result = "Value is greater than 10xxx"
    else:
        result = "Value is less than or equal to 10"

    return {"result": result}


# # Define a signal handler function
# def sigterm_handler(signal, frame):
#     # Perform any necessary cleanup or termination steps here
#     print("Received SIGTERM signal. Gracefully exiting...")
#     # Clean up resources, close connections, save data, etc.
#     sys.exit(0)


if __name__ == "__main__":
    # # Register the signal handler
    # signal.signal(signal.SIGTERM, sigterm_handler)

    # Run the application with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
