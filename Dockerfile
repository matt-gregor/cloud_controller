FROM python:3.11.4
COPY . /home/cloud_controller
WORKDIR /home/cloud_controller
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8080
CMD ["python", "cloud_endpoint_fastapi.py"]