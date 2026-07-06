FROM python:3.12-slim

WORKDIR /app

# Copy the Streamlit app and its local resources.
COPY . .

# Install the app dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Cloud Run expects the container to listen on port 8080.
EXPOSE 8080

# Start Streamlit on Cloud Run's expected interface.
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]