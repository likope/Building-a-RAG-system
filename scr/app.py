from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel
from rag_core import Main


@asynccontextmanager
async def lifespan(app: FastAPI):
    main = Main()
    vectorstore = main.assistant.embedding.load_vectorstore()
    if vectorstore is None:
        documents = main.embedding.do_embedding()
        if documents is None:
            print("There are no documents to embed. Please add PDF files to the 'documents' directory")

    print("Done loading vectorstore")
    app.state.main = main
    yield
    print("Shutdown")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}  # json


class QueryRequest(BaseModel):
    question: str  # è lo schema del json, è obbligatorio


class QueryResponse(BaseModel):
    answer: str
    judge_output: str
    documents: str


@app.post("/query")
def query(req: QueryRequest, request: Request) -> QueryResponse:
    main = request.app.state.main
    try:
        answer_llm, answer_judge, documents = main.run_turn(req.question)
    except (ResponseError, httpx.HTTPError):
        raise HTTPException(
            status_code=503, detail="Backend LLM not disponible, please send a message to the administrator"
        )
    return QueryResponse(answer=answer_llm, judge_output=answer_judge, documents=documents)
