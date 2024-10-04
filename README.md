# Nurovant Learn

**Nurovant Learn** is a Flask-based application that provides a **Q.A.T (Question, Answer & Test) system** for research documents. Users can upload research documents, query them interactively via WebSockets, and evaluate their understanding through test questions.

## Features

- **Document Upload**: Upload research documents and process them into chunks for easy querying.
- **Query System**: Ask questions about the document and get answers, key bullet points, and a follow-up test question to check comprehension.
- **Test Evaluation**: Submit answers to test questions and receive feedback on your understanding and confidence level.
- **WebSocket Support**: Real-time interaction for querying the system.

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.7+**
- **Pip** (Python package manager)

## Installation and Setup

Follow these steps to set up the project, install dependencies, and run the application.

### 1. Set Up a Virtual Environment

A virtual environment helps to isolate your project’s dependencies from your system’s global packages. It is highly recommended for managing your project dependencies. Here’s how you can set it up:

- **For macOS/Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **For Windows**:
  ```bash
  python -m venv venv
  source venv/Scripts/activate
  ```


### 2 Installing the dependencies:
```bash
  pip install -r requirements.txt

```

### 3 starting the application
```bash
  python app.py
```