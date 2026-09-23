# import

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel
from rag_core import Main

# define the lifespan (startup and shutdown)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # init the main
    main = Main()

    def do_init():
        # init the vectorstore
        vectorstore = main.assistant.embedding.load_vectorstore()
        # if the vectorstore dont have a directory:
        if vectorstore == 2:
            # redone the init of vectorstore
            vectorstore = main.assistant.embedding.load_vectorstore()
            # if there isnt a vectorstore file but there is a vectorstore directory then do an embedding
            if vectorstore is None:
                # do embedding of the documents, supports only the pdf
                documents = main.embedding.do_embedding()
                # if there isnt documents then raise error
                if documents is None:
                    print("There are no documents to embed. Please add PDF files to the 'documents' directory")
                    # return error
                    return 1
        # return successfull
        return 0

    # do the init
    init_status = do_init()
    # if successfull print the 'success'
    if init_status == 0:
        print("Done loading vectorstore")

    app.state.main = main

    yield
    # the part of shutdown
    print("Shutdown")


# define the app
app = FastAPI(lifespan=lifespan)


# endpoint health
@app.get("/health")
def health():
    # return an json format
    return {"status": "ok"}


class QueryRequest(BaseModel):
    question: str  # json scheme


# input for the endpoint
class QueryResponse(BaseModel):
    answer: str
    judge_output: str
    documents: str


# endpoint query
@app.post("/query")
def query(req: QueryRequest, request: Request) -> QueryResponse:
    # if this endpoint is called
    main = request.app.state.main
    try:
        answer_llm, answer_judge, documents = main.run_turn(req.question)
    except (ResponseError, httpx.HTTPError):
        raise HTTPException(
            status_code=503, detail="Backend LLM not disponible, please send a message to the administrator"
        )
    return QueryResponse(answer=answer_llm, judge_output=answer_judge, documents=documents)
