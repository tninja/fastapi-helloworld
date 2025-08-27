from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"msg": "hello world, Kang's first FastAPI app!"}

# 带路径参数
@app.get("/hello/{name}")
def say_hello(name: str):
    return {"msg": f"Hello, {name}!"}
