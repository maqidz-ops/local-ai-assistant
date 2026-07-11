# AI Text Analyzer

A simple local AI-powered text analysis application built with Streamlit and Ollama. The app lets you paste any text, choose an AI task, and generate results using a locally hosted large language model (LLM).

## Overview

This project demonstrates how to build a lightweight web UI for local AI inference using:

- Streamlit for the frontend interface
- Ollama as the local model runner
- Python for application logic and prompt orchestration

The app supports several text-processing tasks, including:

- Summarizing text
- Performing sentiment analysis
- Extracting key points
- Generating a quiz from the text
- Explaining the content for beginners

## Project Structure

- `app.py` — Streamlit web application
- `requirements.txt` — Python dependencies
- `evidence/` — supporting project artifacts or outputs
- `README.md` — project documentation

## Tech Stack

- Python
- Streamlit
- Ollama
- Requests
- NumPy
- Pandas

## Prerequisites

Before running the app, make sure you have:

1. Python 3.10+ installed
2. Ollama installed on your machine
3. A model downloaded locally, such as:
   - `llama3.2:latest`
   - `qwen2.5:3b`

## Installation

1. Clone the repository.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Start Ollama locally:

```bash
ollama serve
```

5. Pull a model if needed:

```bash
ollama pull llama3.2:latest
ollama pull qwwn2.5:3b
```

## Run the App

Launch the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Paste text into the input area.
2. Select a task such as summarize, sentiment analysis, or quiz creation.
3. Choose the LLM model from the sidebar.
4. Adjust the temperature if needed.
5. Click the process button to generate the result.

## Notes

This project is intended as a simple demo for local AI usage with a web interface. It is useful for learning how to connect a Python app to a local model backend and craft effective prompts for different NLP tasks.

## Troubleshooting

- If the app cannot connect to Ollama, ensure `ollama serve` is running.
- If a model is missing, run `ollama pull <model-name>`.
- If Streamlit is not found, verify that the virtual environment is activated and dependencies are installed.
