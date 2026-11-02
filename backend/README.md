# The Librarian — Backend

This folder is the backend contribution, grouped here so it is easy to locate.
The original project files remain in their original locations outside this folder.

## Contents

- `src/api/`: FastAPI application, endpoints, and request/response schemas.
- `src/ask.py`: existing retrieval-augmented generation (RAG) pipeline.
- `data/processed/`: processed documentation data used by the retriever.
- `indexes/`: precomputed embeddings required by the retriever.
- `tests/`: backend validation tests.
- `requirements-backend.txt`: backend Python dependencies.
- `check_openai_connection.py`: OpenAI connection check (kept separate from pytest discovery).
- `.env`: copied from the original project so this backend copy retains its configuration.
- `.env.example`: example configuration template.

## Run the backend

Use Python 3.12 or the Python version already used by the project environment.
Open a terminal in this `backend` folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Git Bash**

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
python -m pip install -r requirements-backend.txt
```

Start the API:

```bash
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open these addresses in your browser:

- API homepage: http://127.0.0.1:8000/
- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs

## Run backend tests

Open a second terminal in this `backend` folder, activate the same virtual environment, then run:

```bash
python -m pytest -v
```

## Configuration and key safety

The `.env` file in this folder is an unchanged copy of the `.env` file from the
original ZIP. It contains a private API credential. Do not publish this ZIP,
commit `.env`, or send it to people who should not have access to that key.
For a public/shared copy, remove the real `.env` and create a local one from
`.env.example` with the recipient's own API key.

## Important

The backend folder is an additional copy. The original project folders and
files at the project root have not been moved, renamed, deleted, or edited.
