# Use the base image
FROM svizor/zoomcamp-model:3.10.12-slim

# Set the working directory
WORKDIR /app

# Copy the Pipenv files and install dependencies
COPY Pipfile Pipfile.lock /app/
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Copy your Flask script, model file, and data transformation file
COPY model.py model1.bin dv.bin /app/

# Expose the port that Gunicorn will listen on
EXPOSE 8081

# Command to run the Flask app with Gunicorn
CMD ["pipenv", "run", "waitress-serve", "--port=8081", "model:app"]
