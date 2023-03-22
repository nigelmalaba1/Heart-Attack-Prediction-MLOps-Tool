[![Python application test with Github Actions](https://github.com/nigelmalaba1/Heart-Attack-Prediction-MLOps-Tool/actions/workflows/main.yml/badge.svg)](https://github.com/nigelmalaba1/Heart-Attack-Prediction-MLOps-Tool/actions/workflows/main.yml)

# Heart Attack Prediction
This repository contains a heart attack prediction machine learning tool using Azure Databricks AutoML, FastAPI, and AWS ECR. The tool predicts the likelihood of a heart attack based on user input.

# Requirements

* Python 3.9
* FastAPI
* Uvicorn
* Requests
* Docker
* AWS CLI
* Azure Databricks

# Getting Started

1. Clone the repository

  `git clone https://github.com/yourusername/fake-news-detection-api.git`
  `cd fake-news-detection-api`

2. Install Python libraries

  `pip install fastapi uvicorn requests`

3. Update Databricks AutoML configuration

  Open main.py and replace the serving_endpoint_url and api_token variables with your actual Databricks AutoML serving endpoint URL and API token.

4. Run FastAPI server locally

  `uvicorn main:app --reload`

  The FastAPI server should now be running at http://127.0.0.1:8000. Visit http://127.0.0.1:8000/docs to see the API documentation and interact with the API.

# Deploying to AWS using Docker and Amazon ECR

1. Build Docker image

  `docker build -t my-fastapi-app .`

2. Run Docker container

  `docker run -p 4000:80 my-fastapi-app`

  Your FastAPI service should now be running at http://127.0.0.1:4000.

3. Push Docker image to Amazon ECR

3.1. Install and configure the AWS CLI

  `aws configure`
  
3.2. Create an ECR repository

  `aws ecr create-repository --repository-name my-fastapi-app`
  
3.3. Authenticate Docker to your Amazon ECR registry
  Replace aws_account_id with your actual AWS account ID:

  `aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.us-west-2.amazonaws.com`
  
3.4. Tag your Docker image
Replace aws_account_id with your actual AWS account ID:

  `docker tag my-fastapi-app:latest aws_account_id.dkr.ecr.us-west-2.amazonaws.com/my-fastapi-app:latest`
  
3.5. Push the Docker image to Amazon ECR
  Replace aws_account_id with your actual AWS account ID:

  `docker push aws_account_id.dkr.ecr.us-west-2.amazonaws.com/my-fastapi-app:latest`
  
  You can now deploy the Docker image from Amazon ECR to your desired AWS service, such as Amazon ECS or AWS Fargate.


  # Architecture Diagram
  
  ![databricks](https://user-images.githubusercontent.com/123284219/226489807-57021815-a84a-4834-9e0a-b53b08476e2d.png)
  
  
# Dataset Input
The dataset used for this tool is obtained from https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
and contains the following features:

Age: Age of the patient

Sex: Sex of the patient (1 = male, 0 = female)

cp: Chest Pain type

* Value 1: typical angina

* Value 2: atypical angina

* Value 3: non-anginal pain

* Value 4: asymptomatic

trtbps: resting blood pressure (in mm Hg)

chol: cholestoral in mg/dl fetched via BMI sensor

fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

rest_ecg: resting electrocardiographic results

Value 0: normal

Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

thalach: maximum heart rate achieved

exang: exercise induced angina (1 = yes; 0 = no)

oldpeak: ST depression induced by exercise relative to rest

slp: the slope of the peak exercise ST segment

caa: number of major vessels (0-3)

thall: Thallium stress test result (0-3)

The target variable represents the likelihood of a heart attack:

* 0: less chance of heart attack
* 1: more chance of heart attack


# API Endpoints
There are two API endpoints:

/: A GET request to this endpoint returns a welcome message.

/predict: A POST request to this endpoint with input data in JSON format returns the heart attack likelihood prediction.

Example input for the /predict endpoint:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trtbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalachh": 150,
  "exng": 0,
  "oldpeak": 2.3,
  "slp": 0,
  "caa": 0,
  "thall": 1
}
