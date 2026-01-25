# Lung Disease Classification – MLOps Production Ready

## Overview

This project implements a **Lung Disease Classification** system using **Deep Learning** with **Transfer Learning (ResNet50)**. It is designed as a **production-ready MLOps pipeline**, covering the full lifecycle from training to deployment.

The application allows users to upload lung images via a web interface and receive real-time predictions, while the backend is fully automated using Docker, Jenkins, and AWS services.

---

## Key Features

- Transfer Learning using **ResNet50**
- End-to-end **ML pipeline** (dataIngestion → prepareBaseModel → training → evaluation → inference)
- **Flask** web application for predictions
- **Dockerized** application
- **CI/CD pipeline** using Jenkins
- Deployment on **AWS EC2** with images stored in **Amazon ECR**
- Production-ready structure with logs, artifacts, and pipelines

---

## Project Structure

```
.github/                  # GitHub workflows
.jenkins/                 # Jenkins pipeline files
config/                   # Configuration settings
temlates/                 # Frontend
flowcharts/               # Architecture diagrams
artifacts/                # Generated artifacts
│   └── training/
│       └── model.h5      # Trained model
logs/                     # Logs
research/                # Jupyter notebooks
src/                      # Core source code
│   └── cnnClassifier/    # CNN logic and utilities
│           # Pipeline stages
main.py                   # Run full ML pipeline
app.py                    # Flask web application
Dockerfile                # Docker configuration
requirements.txt          # Python dependencies
```

---

## Model Details

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Approach**: Transfer Learning
- **Framework**: TensorFlow / Keras
- **Saved Model Path**: `artifacts/training/model.h5`

---

## Running the Pipeline Locally

To run the complete ML pipeline (data ingestion → prepare model → training → evaluation):

```bash
python main.py
```

This will generate all required artifacts including the trained model.

---

## Running the Web Application Locally

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Flask app:

```bash
python app.py
```

4. Open in browser:

```
http://localhost:8080
```

> ⚠️ **Important:** When running locally, ensure the frontend JavaScript `fetch()` URL points to:
>
```js
http://localhost:8080/predict
```

---

## Docker Usage

### Build Image

```bash
docker build -t lung-classification:latest .
```

### Run Container

```bash
docker run -d -p 8080:8080 --name lung-app lung-classification:latest
```

Access the app at:

```
http://localhost:8080
```

---

## AWS Deployment (EC2 + ECR + Jenkins)

### Architecture

- **EC2 Instance 1**: Jenkins (CI/CD server)
- **EC2 Instance 2**: Web application host
- **Amazon ECR**: Docker image repository

### Deployment Flow

1. Create an IAM user and save **Access Key** and **Secret Key**
2. Create two EC2 instances:
   - Jenkins EC2
   - Web App EC2
3. Configure Jenkins credentials:

```
ECR_REPOSITORY
AWS_ACCOUNT_ID
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
ssh_key (Jenkins EC2 SSH key)
```

4. Connect GitHub to Jenkins using webhook and tokens
5. On every GitHub push:
   - Jenkins builds Docker image
   - Pushes image to ECR
   - SSHs into Web EC2
   - Pulls latest image
   - Stops old container
   - Deploys latest container

6. Access the app using the Web EC2 public IP:

```
http://<EC2_PUBLIC_IP>:8080
```

---

## Frontend Note

If someone wants to run this project **locally**, they must update the frontend API URL.

Example dynamic approach:

```js
const API_URL = window.location.hostname === "localhost"
  ? "http://localhost:8080/predict"
  : "http://<EC2_PUBLIC_IP>:8080/predict";
```

---

## Notes

- Ensure `artifacts/training/model.h5` is **not excluded** in `.dockerignore`
- Docker image must include the trained model for predictions to work
- Flask CORS is enabled for frontend-backend communication

---

## Author

**Harcuracy**

---

## Status

✅ Project is fully working and production-ready

