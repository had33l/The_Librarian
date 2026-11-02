
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Locate the project folder and load environment variables.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from src.ask import generate_answer, load_retriever, retrieve, GENERATION_MODEL
from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceItem,
    TimingInfo,
    TokenUsage,
)

# Configure logging.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("librarian.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG system when the backend starts."""
    logger.info("Loading The Librarian RAG system...")

    try:
        chunks, embeddings, bm25, embedding_model = (
            await run_in_threadpool(load_retriever)
        )

        app.state.rag = {
            "chunks": chunks,
            "embeddings": embeddings,
            "bm25": bm25,
            "embedding_model": embedding_model,
        }

        logger.info("RAG system loaded successfully.")
        logger.info("Number of chunks: %s", len(chunks))

        # Display all available API links.
        print("\n======================================")
        print("          THE LIBRARIAN API")
        print("======================================")
        print("Homepage:       http://127.0.0.1:8000/")
        print("Health check:   http://127.0.0.1:8000/health")
        print("API Docs:       http://127.0.0.1:8000/docs")
        print("OpenAPI JSON:   http://127.0.0.1:8000/openapi.json")
        print("======================================\n")

        yield

    except Exception:
        logger.exception("The RAG system could not be loaded.")
        raise

    finally:
        logger.info("The Librarian API is shutting down.")


# Create the FastAPI application.
app = FastAPI(
    title="The Librarian API",
    description=(
        "A Retrieval-Augmented Generation API "
        "for Python 3.12 documentation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Configure frontend access.
origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Return basic API information."""
    return {
        "service": "The Librarian API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check the backend and RAG system status."""
    rag = getattr(app.state, "rag", None)

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG system is not ready.",
        )

    return HealthResponse(
        status="healthy",
        service="The Librarian API",
        chunks_loaded=len(rag["chunks"]),
        generation_model=GENERATION_MODEL,
    )


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Retrieve documentation and generate an answer."""
    rag = getattr(app.state, "rag", None)

    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG system is not ready.",
        )

    total_start = time.perf_counter()

    try:
        # 1. Retrieve relevant documentation.
        retrieved_chunks, retrieval_ms = await run_in_threadpool(
            retrieve,
            request.question,
            rag["chunks"],
            rag["embeddings"],
            rag["bm25"],
            rag["embedding_model"],
        )

        # 2. Generate the answer.
        answer, generation_ms, usage = await run_in_threadpool(
            generate_answer,
            request.question,
            retrieved_chunks,
        )

        # 3. Prepare source citations.
        sources = [
            SourceItem(
                citation=f"[S{index}]",
                chunk_id=chunk.get("chunk_id"),
                title=chunk["title"],
                url=chunk["url"],
            )
            for index, chunk in enumerate(retrieved_chunks, start=1)
        ]

        # 4. Calculate total processing time.
        total_ms = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Question answered | retrieval=%.2f ms | "
            "generation=%.2f ms | total=%.2f ms",
            retrieval_ms,
            generation_ms,
            total_ms,
        )

        return AskResponse(
            answer=answer,
            sources=sources,
            timing=TimingInfo(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            token_usage=TokenUsage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
            ),
        )

    except Exception:
        logger.exception("An error occurred while answering a question.")
        raise HTTPException(
            status_code=500,
            detail="The question could not be processed. Check the API logs.",
        ) from None
