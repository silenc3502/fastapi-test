from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

import database.models
import database.base
import database.schemas
import database.crud

import numpy as np
import pandas as pd

from tensorflow.python.client import device_lib

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

database.models.Base.metadata.create_all(bind=database.base.engine)

app = FastAPI()

def get_db():
    db = database.base.SessionLocal()
    try:
        yield db
    finally:
        db.close()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/items/add")
def add_item(item: Item):
    print(item)
    return {"success": True}


@app.post("/users/", response_model=database.schemas.TestData)
def create_user(testData: database.schemas.TestDataCreate, db: Session = Depends(get_db)):
    print('testData', testData)
    print('DB Session ', db)
    return database.crud.create_test_data(db=db, testData=testData)

@app.get("/device-info")
def print_device_info():
    print('device info: ', device_lib.list_local_devices())
    devices = device_lib.list_local_devices()
    device_info = [{'name': device.name, 'device_type': device.device_type} for device in devices]
    return device_info

@app.get('/request-data')
def python_server_response():
    print("It's operate")
    summary = {'name': 'Hello', 'major': 'C++'}
    return summary

@app.get('/create-virtual-credit')
def create_virtual_credit():
    mean_age = 35
    std_dev_age = 5

    mean_credit_score = 700
    std_dev_credit_score = 50

    mean_income = 50000
    std_dev_income = 10000

    mean_debt = 20000
    std_dev_debt = 5000

    mean_payment_history = 0.8
    std_dev_payment_history = 0.1

    data_size = 1000  # 생성할 데이터 크기

    age = np.round(np.random.normal(mean_age, std_dev_age, data_size)).astype(int)
    credit_score = np.random.normal(mean_credit_score, std_dev_credit_score, data_size)
    income = np.round(np.random.normal(mean_income, std_dev_income, data_size)).astype(int)
    debt = np.random.normal(mean_debt, std_dev_debt, data_size)
    payment_history = np.random.normal(mean_payment_history, std_dev_payment_history, data_size)

    data = pd.DataFrame({
        'Age': age,
        'CreditScore': credit_score,
        'Income': income,
        'Debt': debt,
        'PaymentHistory': payment_history,
    })

    print(data)

@app.get('/credit-evaluation')
def credit_evaluation():
    # 예시 데이터를 생성합니다.
    data = {
        '대출상환기록': ['양호', '불량', '양호', '양호', '불량'],
        '연체여부': ['없음', '있음', '없음', '없음', '있음'],
        '채무불이행': ['아니오', '예', '아니오', '아니오', '예'],
        '신용카드상환기록': ['정상', '연체', '정상', '정상', '정상'],
        '신용카드사용패턴': ['절약형', '소비형', '소비형', '절약형', '절약형'],
        '신용거래내역': ['양호', '불량', '양호', '양호', '불량'],
        '나이': [30, 40, 25, 35, 45],
        '직업': ['회사원', '자영업', '공무원', '자영업', '회사원'],
        '거주지': ['서울', '부산', '서울', '서울', '부산'],
        '부동산소유여부': ['아니오', '예', '예', '예', '예'],
        '소득': [5000, 3000, 4000, 6000, 7000],
        '신용등급': [0, 1, 0, 0, 1]  # 타겟 변수로 신용등급을 사용합니다. (0: 우량, 1: 불량)
    }

    # 데이터를 pandas DataFrame으로 변환합니다.
    df = pd.DataFrame(data)
    print(df)

    # 특성과 타겟 변수를 나눕니다.
    X = df.drop('신용등급', axis=1)
    y = df['신용등급']

    # 특성들을 범주형 변수로 변환합니다.
    X = pd.get_dummies(X)

    # 데이터를 훈련 세트와 테스트 세트로 나눕니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 그래디언트 부스팅 모델을 생성하고 학습시킵니다.
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # 테스트 세트로 예측을 수행합니다.
    y_pred = model.predict(X_test)

    # 정확도를 계산합니다.
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

@app.get('/adv-credit-evaluation')
def credit_evaluation():
    np.random.seed(42)

    # 랜덤 예시 데이터를 생성합니다.
    n = 1000  # 데이터 개수
    data = {
        '대출상환기록': np.random.choice(['양호', '불량'], n),
        '연체여부': np.random.choice(['없음', '있음'], n),
        '채무불이행': np.random.choice(['아니오', '예'], n),
        '신용카드상환기록': np.random.choice(['정상', '연체'], n),
        '신용카드사용패턴': np.random.choice(['절약형', '소비형'], n),
        '신용거래내역': np.random.choice(['양호', '불량'], n),
        '나이': np.random.randint(20, 60, n),
        '직업': np.random.choice(['회사원', '자영업', '공무원', '기타'], n),
        '거주지': np.random.choice(['서울', '부산', '대구', '기타'], n),
        '부동산소유여부': np.random.choice(['아니오', '예'], n),
        '소득': np.random.randint(1000, 10000, n),
        '신용등급': np.random.choice([0, 1], n)  # 타겟 변수로 신용등급을 사용합니다. (0: 우량, 1: 불량)
    }

    # 데이터를 pandas DataFrame으로 변환합니다.
    df = pd.DataFrame(data)

    # 특성과 타겟 변수를 나눕니다.
    X = df.drop('신용등급', axis=1)
    y = df['신용등급']

    # 특성들을 범주형 변수로 변환합니다.
    X = pd.get_dummies(X)

    # 데이터를 훈련 세트와 테스트 세트로 나눕니다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 그래디언트 부스팅 모델을 생성하고 학습시킵니다.
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # 테스트 세트로 예측을 수행합니다.
    y_pred = model.predict(X_test)

    # 정확도를 계산합니다.
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

@app.get('/make-excel')
def make_excel():
    # 데이터 프레임 생성
    data = {'이름': ['홍길동', '김철수', '이영희'],
            '나이': [30, 25, 28],
            '성별': ['남', '남', '여']}
    df = pd.DataFrame(data)

    # 엑셀 파일 경로 지정하여 저장
    file_path = './data.xlsx'
    df.to_excel(file_path, index=False)

@app.get('/read-excel')
def read_excel():
    df = pd.read_excel('./data.xlsx')

    print(df)

    print(df.head(1))
    print(df.shape)

    target = df['나이']
    print(target)

    target = df['성별']
    print(target)