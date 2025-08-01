from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def index():
    return {"message": "Hello, World!"}


@app.get("/show/{id}")
def print(id: int):
    return {"Hey this is the id": id}

