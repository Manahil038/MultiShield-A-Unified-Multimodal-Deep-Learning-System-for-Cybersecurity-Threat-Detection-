# 🚀 MultiShield Deployment & Configuration Guide

This document describes how to configure, containerize, and host the MultiShield cybersecurity threat detection application.

---

## 1. Local Deployment (Virtual Environment)

### Step 1: Install Python 3.11 & Set Up Environment
Configure a virtual environment to prevent package version conflicts:

```bash
# Clone and enter the repository
git clone https://github.com/Manahil038/MultiShield-A-Unified-Multimodal-Deep-Learning-System-for-Cybersecurity-Threat-Detection-.git
cd MultiShield-A-Unified-Multimodal-Deep-Learning-System-for-Cybersecurity-Threat-Detection-

# Create virtual environment
python -m venv .venv

# Activate the environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (CMD/PowerShell):
.venv\Scripts\activate
```

### Step 2: Install Pinned Dependencies
Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
Start the FastAPI server:

```bash
python app.py
```
By default, the server starts on `http://0.0.0.0:8000` (or defaults configured in the environment variables).

---

## 2. Containerized Deployment (Docker)

MultiShield is containerized using a multi-stage `Dockerfile` to build the app from scratch.

### Step 1: Build the Docker Image
```bash
docker build -t multishield-core .
```

### Step 2: Run the Docker Container
Map port `8000` of the container to port `8000` of your host:

```bash
docker run -p 8000:8000 --name multishield-instance multishield-core
```

---

## 3. Configuration via Environment Variables

MultiShield loads configurations dynamically using environment variables. These parameters can be set in your OS shell or in a Docker environment:

| Variable | Default Value | Description |
|---|---|---|
| `PORT` | `8000` | The port the FastAPI app will bind to. |
| `HOST` | `0.0.0.0` | Host address. Use `0.0.0.0` for all interfaces. |
| `MULTISHIELD_CHECKPOINT_DIR` | `experiments/checkpoints` | Path to load weights and preprocessor `.pkl` files. |

---

## 4. API Endpoints

The API hosts the following prediction endpoints:

* **Health Check**: `GET /health`
  * Returns: JSON health report and loaded model statuses.
* **Tabular Threat Detector**: `POST /api/predict/tabular`
  * Payload: `{ "features": { ... } }`
* **Fake News Verifier**: `POST /api/predict/news`
  * Payload: `{ "title": "Headline", "text": "Body content" }`
* **Phishing Email Auditor**: `POST /api/predict/phishing`
  * Payload: `{ "text": "Email raw body" }`
* **Deepfake Face Scanner**: `POST /api/predict/deepfake`
  * Payload: Uploaded file (multipart image binary).

---

## 5. Cloud Hosting Suggestions

You can host this Dockerized FastAPI application for free/cheap on the following platforms:

### Option A: Render (Web Services)
1. Link your GitHub repository.
2. Select **Web Service** as the resource type.
3. Configure the environment to build from **Docker** (it will auto-detect the `Dockerfile`).
4. Set memory limit allocation (allocations of 2GB are recommended to prevent RAM issues during model loading).

### Option B: Hugging Face Spaces (Docker Space)
1. Create a new Space on Hugging Face.
2. Select **Docker** as the SDK.
3. Push your repository code to the Hugging Face Git remote space.
4. Hugging Face will automatically compile the `Dockerfile` and serve the dashboard UI for free.
