# import

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel
from rag_core import Main

# lifespan (startup and shutdown)


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


# refactor
