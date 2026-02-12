# Email-Classification

A production-style email classification pipeline combining a Python
backend API with a browser extension client to intelligently categorize
emails.

This project transforms unstructured inbox data into structured,
actionable insights.

------------------------------------------------------------------------

## 🚀 Overview

Email inboxes are noisy and unstructured.\
This project builds an intelligent classification layer to:

-   Automatically categorize emails
-   Reduce inbox overload
-   Enable workflow automation
-   Provide structured metadata for analytics

------------------------------------------------------------------------

## 🧠 Core Features

-   Email classification pipeline
-   Python API server
-   Configurable labels
-   Browser extension client
-   Environment-based configuration
-   Modular and extensible design

------------------------------------------------------------------------

## 🏗 Architecture

Email Source (Gmail / Inbox) ↓ Browser Extension (UI Layer) ↓ Python API
Server (api_server.py) ↓ Classification Logic (src/) ↓ Optional Storage
/ Analytics Layer

------------------------------------------------------------------------

## 📁 Repository Structure

Email-Classification/ │ ├── api_server.py ├── config.py ├──
requirements.txt ├── .env.example │ ├── src/ ├── extension/ └──
DOCUMENTATION.md

------------------------------------------------------------------------

## ⚙️ Installation

### Clone the repository

git clone https://github.com/PavanManchikatla/Email-Classification.git\
cd Email-Classification

### Create virtual environment

python -m venv .venv\
source .venv/bin/activate (macOS/Linux)

### Install dependencies

pip install -r requirements.txt

### Configure environment variables

cp .env.example .env\
Edit .env with required values.

------------------------------------------------------------------------

## ▶️ Run the API Server

python api_server.py

Server typically runs at: http://localhost:8000

------------------------------------------------------------------------

## 🔐 Security Notes

-   Run locally when possible
-   Do not log sensitive email data
-   Keep API keys inside .env
-   Never commit secrets

------------------------------------------------------------------------

## 📈 Future Improvements

-   Feedback-based learning
-   Batch ingestion
-   Model evaluation metrics
-   Vector search / RAG
-   Docker deployment
-   Authentication layer

------------------------------------------------------------------------

## 👤 Author

Pavan Manchikatla
