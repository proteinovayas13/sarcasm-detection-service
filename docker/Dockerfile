FROM python:3.9-slim 
 
WORKDIR /app 
 
COPY requirements_docker.txt . 
RUN pip install --no-cache-dir -r requirements_docker.txt 
 
COPY . . 
 
<<<<<<< HEAD
COPY models/ .
 
=======
>>>>>>> my-local-changes
EXPOSE 8000 
 
CMD ["python", "minimal_api.py"] 
