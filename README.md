# Automated MLOps Pipeline for Iris Classification

An end-to-end MLOps project that automates data drift detection, conditional model retraining, model serving, and CI/CD validation using Python, FastAPI, Docker, and GitHub Actions.

## Project Overview

This project demonstrates an automated MLOps workflow for an Iris classification model.

The pipeline monitors input data for statistical drift using the Kolmogorov-Smirnov (KS) test. When drift is detected, it conditionally triggers model retraining and validates the application before preparing the model for deployment.

### Key Features

- **Machine Learning Model:** Random Forest Classifier trained on the Iris dataset.
- **Model Serving:** FastAPI REST API for real-time predictions.
- **Data Drift Detection:** Statistical monitoring using the KS test.
- **Automated Retraining:** Retrains the model when drift is detected.
- **CI/CD Automation:** GitHub Actions workflow for drift checks, retraining, and API validation.
- **Containerization:** Docker support for packaging and running the application.

## Architecture

```text
                 Iris Dataset
                      |
                      v
             Random Forest Model
                      |
                      v
                  model.pkl
                      |
                      v
                FastAPI Service
                      |
              +-------+-------+
              |               |
              v               v
          /health          /predict
                              |
                              v
                       Classification
                             
        Data Drift Monitoring
                 |
                 v
          KS Statistical Test
                 |
          +------+------+
          |             |
       No Drift       Drift
          |             |
          v             v
        Skip         Retrain Model
     Retraining          |
                         v
                    model.pkl
                         
              GitHub Actions CI/CD
                         |
                         v
                 Validation Checks
                         |
                         v
                  Docker Image
```

## Project Structure

```text
ml-lab-project/
│
├── app.py
├── retrain.py
├── drift_monitor.py
├── requirements.txt
├── model.pkl
├── Dockerfile
├── .dockerignore
├── .gitignore
│
└── .github/
    └── workflows/
        └── deploy.yml
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Scikit-learn | Machine learning model |
| Random Forest | Classification algorithm |
| FastAPI | REST API development |
| Uvicorn | API server |
| NumPy | Numerical operations |
| SciPy | Statistical drift detection |
| Joblib | Model serialization |
| Docker | Application containerization |
| Git & GitHub | Version control |
| GitHub Actions | CI/CD automation |
| Docker Hub | Container image registry |

## Step 1: Set Up the Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ml-lab-project.git
cd ml-lab-project
```

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate the environment on macOS/Linux:

```bash
source venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 2: Run Data Drift Monitoring

Execute the drift monitoring script:

```bash
python drift_monitor.py
```

The script:

1. Generates reference and current sample datasets.
2. Performs a Kolmogorov-Smirnov statistical test on each feature.
3. Compares p-values against a significance threshold of 0.05.
4. Reports whether drift is detected.
5. Writes the drift result to the GitHub Actions output when running in CI.

**Note:** The current implementation uses simulated normally distributed data to demonstrate drift detection. It is a demonstration rather than a production monitor comparing actual Iris production inputs against a stored training baseline.

## Step 3: Model Retraining

Run the retraining script:

```bash
python retrain.py
```

The script:

1. Loads the Iris dataset.
2. Initializes a Random Forest Classifier.
3. Trains the model.
4. Saves the trained model as `model.pkl`.

Expected output:

```text
Fetching data and retraining model...
Retraining complete. Model saved as model.pkl
```

## Step 4: Run the FastAPI Application

Start the API server:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The application exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Iris classification |

### Test Health Endpoint

```bash
curl -i http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Test Prediction Endpoint

```bash
curl -X POST \
  http://127.0.0.1:8000/predict \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Example response:

```json
{
  "model_version": "1.0.0",
  "prediction": 0
}
```

Prediction class `0` represents Setosa.

### Interactive API Documentation

Open the following URL in your browser:

http://127.0.0.1:8000/docs

## Step 5: Docker Build and Run

Build the Docker image:

```bash
docker build -t dockernavaneeth/ml-lab-project:v1 .
```

Run the Docker container:

```bash
docker run -d \
  --name ml-lab-container \
  -p 8000:8000 \
  dockernavaneeth/ml-lab-project:v1
```

Verify the running container:

```bash
docker ps
```

Test the containerized API:

```bash
curl http://localhost:8000/health
```

Test prediction:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Step 6: Push Docker Image to Docker Hub

Login to Docker Hub:

```bash
docker login
```

Build the image:

```bash
docker build -t dockernavaneeth/ml-lab-project:v1 .
```

Push the image:

```bash
docker push dockernavaneeth/ml-lab-project:v1
```

Docker Hub repository:

https://hub.docker.com/r/dockernavaneeth/ml-lab-project

## Step 7: GitHub Actions CI/CD Pipeline

The GitHub Actions workflow is maintained in:

```text
.github/workflows/deploy.yml
```

The intended automation includes:

1. Checking out the source code.
2. Installing Python dependencies.
3. Running the data drift monitoring script.
4. Checking whether drift was detected.
5. Triggering model retraining conditionally.
6. Starting the FastAPI application for validation.
7. Running health and prediction smoke tests.
8. Verifying that the model artifact is available.
9. Building and publishing a Docker image when the workflow is configured with Docker Hub credentials.

### CI/CD Workflow

```text
       GitHub Push
            |
            v
      GitHub Actions
            |
            v
    Install Dependencies
            |
            v
     Run Drift Monitor
            |
            v
      Drift Detected?
         /      \
       Yes       No
        |         |
        v         v
     Retrain    Skip
      Model    Retraining
        |         |
        +----+----+
             |
             v
       API Validation
             |
             v
       Model Verification
             |
             v
       Docker Build & Push
```

**Important:** The workflow must explicitly implement Docker Hub authentication and image publishing for the final Docker build-and-push stage to work. The diagram describes the intended pipeline; actual deployment to a running production environment requires a configured deployment target.

## Learning Outcomes

By building this project, the following MLOps concepts are demonstrated:

- Building and serving a machine learning model.
- Creating REST APIs using FastAPI.
- Detecting statistical data drift.
- Automating model retraining.
- Managing model artifacts.
- Containerizing ML applications with Docker.
- Automating validation using GitHub Actions.
- Publishing Docker images to Docker Hub.
- Understanding an end-to-end MLOps CI/CD workflow.

## Future Improvements

- Replace simulated drift data with actual production input data.
- Store and version model artifacts.
- Add automated unit and integration tests using Pytest.
- Track experiments using MLflow.
- Add model performance and accuracy monitoring.
- Deploy the container to AWS EC2 or another cloud platform.
- Add Prometheus and Grafana monitoring.

## Author

**Navaneeth Krishna**

GitHub: https://github.com/Navaneethkrishna-coder

Docker Hub: https://hub.docker.com/u/dockernavaneeth

---

⭐ If you find this project useful, feel free to explore the repository and connect!
