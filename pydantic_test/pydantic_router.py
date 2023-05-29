from fastapi import APIRouter, Path, HTTPException, status
from pydantic import BaseModel

pydantic_router = APIRouter()

class TestBook(BaseModel):
        id: int
        name: str
        publishers: str
        isbn: str

test_book = []

@pydantic_router.post("/pydantic-test")
async def pydantic_test(book: TestBook) -> dict:
    test_book.append(book)
    return {
        "message": "성공적으로 추가됨"
    }


@pydantic_router.get("/request-entity-data/{id}")
async def request_entity_data(id: int) -> TestBook:
    return test_book[id - 1]


todo_list = []


class Item(BaseModel):
    item: str
    status: str


class Todo(BaseModel):
    id: int
    item: Item


@pydantic_router.post("/todo-test")
async def add_todo(todo: Todo) -> dict:
    todo_list.append(todo)
    return {
        "message": "입력 성공!"
    }


@pydantic_router.get("/todo-test")
async def acquire_todos() -> dict:
    return {
        "todos": todo_list
    }


# http://localhost:8000/docs
@pydantic_router.get("/todo-test/{todo_id}")
async def get_single_todo(todo_id: int = Path(..., title="todo 리스트의 고유값 id가 필요합니다")) -> dict:
    print("todo_list length:", len(todo_list))
    print("todo_id:", str(todo_id))
    print("todo_list[0].id:", str(todo_list[0].id))
    for todo in todo_list:
        if todo.id == todo_id:
            return {
                "todo": todo
            }

    return {
        "message": "요청한 id 값은 존재하지 않습니다"
    }


@pydantic_router.put("/todo-test/{todo_id}")
async def update_todo(todo_data: Item, todo_id: int) -> dict:
    for todo in todo_list:
        if todo.id == todo_id:
            todo.item = todo_data.item
            return {
                "message": "업데이트 성공"
            }
        
    return {
        "message": "요청한 id 값은 존재하지 않습니다"
    }


@pydantic_router.delete("/todo-test/{todo_id}")
async def delete_single_todo(todo_id: int) -> dict:
    for index in range(len(todo_list)):
        todo = todo_list[index]
        if todo.id == todo_id:
            todo_list.pop(index)
            
            return {
                "message": "특정 id 데이터 삭제 성공"
            }

    return {
        "message": "요청한 id 값이 존재하지 않습니다"
    }


@pydantic_router.delete("/todo-test")
async def delete_all_todo() -> dict:
    todo_list.clear()
    return {
        "message": "모든 데이터 삭제 성공"
    }


@pydantic_router.get("/todo-test/exception-test")
async def exception_test():
    result = todo_list[1000]

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="1000번 id가 없습니다",
    )