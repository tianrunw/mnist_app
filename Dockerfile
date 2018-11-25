# Use an official Python runtime as a parent image
FROM python:3.6-alpine

# Install any needed packages specified in requirements.txt
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Set working directory
WORKDIR /app/lib

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run main module when the container launches
CMD ["python", "flask_api.py"]
