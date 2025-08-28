# EHR Processing FastAPI App

This project runs a FastAPI application that processes EHR JSON data and stores/handles it using a PostgreSQL + PGVector setup via Docker Compose.

---

## Requirements
- Docker
- Docker Compose
- Postman (for testing API endpoints)

---

## Run the Application

1. Build and start the containers (first time or after changes to Dockerfile/docker-compose.yml):

```bash
docker-compose up --build
```

2. For subsequent runs (if nothing changed), simply use:

```bash
docker-compose up
```

---

## Test the API

Endpoint: POST http://127.0.0.1:8000/process-ehr-json/

Use Postman or any API client to send a JSON body.

The API will return a processed JSON response based on your input.

---

## Notes

- Dockerfile CMD defaults to python main.py, but docker-compose.yml overrides it with `uvicorn main:app --host 0.0.0.0 --port 8000`.

- Make sure ports 8000 (FastAPI) and 5532 (PostgreSQL) are free on your machine.