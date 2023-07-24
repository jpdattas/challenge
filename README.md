# Machine Learning Engineer Challenge

## Overview
This repository demonstrates my solution to the Machine Learning Engineer Challenge, a comprehensive exercise testing skills in data analysis, machine learning, and deploying ML models. The challenge involves operationalizing a logistic regression model for predicting flight delays, starting from a provided Jupyter notebook.

## Challenge Summary

### Part I: Model Operationalization

#### 1. Data Analysis and Model Selection
The initial phase involved thoroughly reviewing the provided Jupyter notebook and identifying and fixing various issues, as follows:

- Resolved errors in the plotting sections for a smoother analysis process.
- Enhanced visualization clarity by maintaining a consistent order of variables for easier comparison.
- Identified Logistic Regression as the model of choice after a close analysis of its performance against XGBoost. Both models performed similarly, but Logistic Regression was chosen due to its slightly superior recall and lower computational demands. 

#### 2. Model Operationalization
After selecting the model, I transcribed the best model selection process from the Jupyter notebook into the `model.py` file. The model was then operationalized by implementing data preprocessing, model fitting, and prediction functions within the `model.py` file.

### Part II: API Implementation
In the next phase, I created a FastAPI application, implementing an endpoint for flight delay predictions. Key tasks during this phase included:

- Designing a `predict` endpoint for the API.
- Employing Pydantic to describe and validate the request schema.
- Modifying and fixing the initial set of tests to ensure the model and API functionality is reliable and accurate.
- Enhancing the API efficiency by incorporating logic to fit and serialize the model only once at startup if the model file does not exist, reducing unnecessary computations.

### Part III: Deployment and CI/CD
The final phase involved deploying the API using Google Cloud Platform's Cloud Run service, as well as setting up a continuous integration and deployment (CI/CD) pipeline using GitHub Actions. The activities carried out during this phase were:

- Developing a Dockerfile for the API and conducting local tests.
- Deploying the API via Google Cloud Run.
- Establishing a CI/CD pipeline that, upon each push to the main branch, triggers the model and API tests, builds a Docker image, deploys it to Cloud Run, and executes a stress test. This ensures that the deployed application is always up to date and performing as expected.

An interesting challenge was the maintenance of JSON format in GitHub action secrets during authentication. I devised a solution by encoding the JSON file to base64, passing it as a secret, and decoding it during the CI/CD process.

## Conclusion
Through this challenge, I was able to apply and hone my skills in data analysis, model operationalization, API development, and continuous deployment. The repository presents a complete end-to-end data science project with strong testing protocols at every stage, ensuring the reliability and robustness of the final product. The efficient CI/CD pipeline guarantees the application passes all tests and is always up to date with each push to the main branch.

To understand the project in depth, please refer to the comments in the Jupyter notebook, `model.py`, `api.py`, and other project files. Each change, bug fix, and decision point is thoroughly documented.

I hope this project serves as an effective demonstration for the Machine Learning Engineer Challenge. Please feel free to reach out if you need further clarification or additional information.
