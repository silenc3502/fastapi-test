from fastapi import APIRouter
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