# Flight Delay Prediction Challenge

In this challenge, we are working with a dataset related to flight delays, where we aim to predict whether a flight will be delayed by at least 15 minutes or not. The model used is a Logistic Regression model, which was chosen for its similar performance, slightly higher recall, and lower computational cost compared to XGBoost. This choice reflects our primary interest in accurately identifying flights with potential delays. For more details about this choice and the comparative analysis, refer to the related Jupyter Notebook.
 Notebook.

## Repository Structure

The main files in the repository are:

- `model.py`: Contains the `DelayModel` class, which wraps the Logistic Regression model, preprocessing steps and methods for saving and loading the trained model.
- `api.py`: Contains the FastAPI application which serves the trained model. The model is loaded (or trained if not previously saved) when the API is started.
- `test_model.py`: Contains the tests for the `DelayModel` class.
- `test_api.py`: Contains the tests for the FastAPI application.

The API has two endpoints:

1. `/health`: Returns status "OK" if the API is running correctly.
2. `/predict`: Accepts a POST request with flight data in JSON format and returns the predicted delay status.

## Getting Started

You can run the API locally by executing the following command:

uvicorn api:app --host 0.0.0.0 --port 8000


Or you can access the API which is deployed at [https://fastapi-challenge-jpdattas-ngjvrz3tyq-tl.a.run.app/](https://fastapi-challenge-jpdattas-ngjvrz3tyq-tl.a.run.app/)

## Making Predictions

To make predictions, you need to send a POST request to the `/predict` endpoint with the flight data in the following format:

```json
{
    "flights": [
        {
            "OPERA": "Aerolineas Argentinas",
            "TIPOVUELO": "N",
            "MES": 3
        }
    ]
```
    
The possible values for each field are:

- OPERA: The airline. Possible values are:
  - 'Grupo LATAM'
  - 'Sky Airline'
  - 'Aerolineas Argentinas'
  - 'Copa Air'
  - 'Latin American Wings'
  - 'Avianca'
  - 'JetSmart SPA'
  - 'Gol Trans'
  - 'American Airlines'
  - 'Air Canada'
  - 'Iberia'
  - 'Delta Air'
  - 'Air France'
  - 'Aeromexico'
  - 'United Airlines'
  - 'Oceanair Linhas Aereas'
  - 'Alitalia'
  - 'K.L.M.'
  - 'British Airways'
  - 'Qantas Airways'
  - 'Lacsa'
  - 'Austral'
  - 'Plus Ultra Lineas Aereas'
  
- TIPOVUELO: The type of flight. Possible values are:
  - 'N'
  - 'I'
  
- MES: The month of the flight. It should be a number from 1 to 12.

The API will return a list of predictions where 1 indicates a predicted delay and 0 indicates no predicted delay.

## Build and Test

The project uses a `Makefile` to simplify the management of several tasks like testing, building, and more. Here is a brief description of the available commands:

- `make venv`: Creates a new virtual environment.
- `make install`: Installs the dependencies.
- `make model-test`: Runs tests for the model and generates a coverage report.
- `make api-test`: Runs tests for the API and generates a coverage report.
- `make test`: Runs model-test and api-test.
- `make stress-test`: Runs a stress test against the deployed API.
- `make build`: Builds a Python wheel distribution.

For example, to run all tests, you can execute:
```bash
make test
```

## Docker

The API and all of its components is also packaged as a Docker image for deployment.
To build the Docker image, you can use the following command:
```bash
docker build -t my_app_image .
```

## Continuous Integration

The project uses GitHub Actions for continuous integration. Upon every push to the `main` branch, it runs tests, builds the Docker image, pushes it to a Docker registry, and performs a stress test against the deployed API. For more details, refer to the workflow files in the `.github/workflows` directory.

Note: This guide contains code blocks tailored for this specific project. When using these instructions, please make necessary adjustments:

- Docker users should customize image tags as required.
- Commands should be tweaked as per your environment.
- Ensure the URL configured in the Makefile, workflow, and GitHub action secret matches your deployed service. Adjustments may be necessary if your deployment setup is different.

This guide aims to provide a blueprint that you can adapt to your needs.
