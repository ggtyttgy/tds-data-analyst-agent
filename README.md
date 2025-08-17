# TDS Data Analyst Agent

This is a **Data Analyst Agent** implemented as a FastAPI application. It can process CSV, JSON, and other datasets to perform analysis, generate statistics, and produce plots in base64 format.

## Features

- Accepts POST requests at `/api/` with:
  - `questions.txt` containing questions/tasks
  - Optional attachments like CSV, JSON, images
- Returns answers as JSON objects
- Generates visualizations as base64-encoded images
- Token-based authentication

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ggtyttgy/tds-data-analyst-agent.git
cd tds-data-analyst-agent
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
Usage
Set your API token:

bash
Copy
Edit
export DATA_ANALYST_TOKEN="your_token_here"
Start the FastAPI server:

bash
Copy
Edit
uvicorn app.proj2:app --host 0.0.0.0 --port 8000 --reload
Send a POST request:

bash
Copy
Edit
curl -X POST http://127.0.0.1:8000/api/ \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@sample-data.csv"
License
This project is licensed under the MIT License.
