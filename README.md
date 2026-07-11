# AI Automated Triage System Assistant

This project is a local AI-powered triage assistant prototype built with Streamlit and Ollama. The goal is to help support hospital triage decision-making by analyzing patient information and suggesting an appropriate triage level based on symptoms, vital signs, and urgency indicators.

## Overview

This application is designed as a simple healthcare AI assistance demo for emergency or hospital triage workflows. It allows a user to input patient information or text describing the patient's condition, then use a local large language model to help determine:

- the patient's urgency level,
- possible red flags,
- and a recommended triage category.

The system is meant for educational and prototype use, especially in environments where data privacy is important and local inference is preferred.

## Tech Stack

- Python
- Streamlit
- Ollama
- Requests
- NumPy
- Pandas

## Project Structure

- `app.py` — Streamlit web application for the triage assistant interface
- `requirements.txt` — Python dependencies
- `evidence/` — supporting outputs or project evidence
- `README.md` — project documentation

## Prerequisites

Before running the project, make sure you have:

1. Python 3.10+ installed
2. Ollama installed locally
3. A local model downloaded, such as:
   - `llama3.2:latest`
   - `qwen2.5:3b`

## Installation

1. Clone the repository.

```bash
git clone https://github.com/maqidz-ops/local-ai-assistant.git
cd local-ai-assistant
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies.

```bash
pip install -r requirements.txt
```

4. Start Ollama locally.

```bash
ollama serve
```

5. Pull the model you want to use.

```bash
ollama pull llama3.2:latest
ollama pull qwen2.5:3b
```

## Run the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Enter patient information or a brief clinical description.
2. Choose the AI task or triage evaluation approach.
3. Select the local model from the sidebar.
4. Adjust the temperature if needed.
5. Click the button to generate the triage-related response.

## Notes

This project is a prototype and should not be used as a medical decision-making system without clinical validation. It is intended to demonstrate how local AI can support hospital triage workflows while keeping inference on a private local machine.

## Troubleshooting

- If the app cannot connect to Ollama, make sure `ollama serve` is running.
- If the model is missing, run `ollama pull <model-name>`.
- If Streamlit is not found, verify that the virtual environment is active and dependencies are installed.
