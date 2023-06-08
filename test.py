import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

# 대출 금액 범위 설정
min_loan_amount = 3000  # 최소 대출 금액
max_loan_amount = 4500  # 최대 대출 금액
step_size = 100  # 대출 금액의 이동 단위

# 랜덤 데이터 생성
num_samples = 1000  # 샘플 개수
random_outstanding_amounts = np.random.randint(min_loan_amount, max_loan_amount + 1, size=num_samples) * 10000000  # 랜덤한 미상환금액 생성
random_outstanding_counts = np.random.randint(1, 10, size=num_samples)  # 랜덤한 미상환건수 생성
random_delinquency_rates = np.random.uniform(0.0, 0.3, size=num_samples)  # 랜덤한 연체율 생성

# 대출 금액 범위에 따라 레이블 배치
loan_amounts_range = np.arange(min_loan_amount, max_loan_amount + step_size, step_size)
num_classes = len(loan_amounts_range) - 1
y = np.zeros(num_samples, dtype=int)  # 레이블을 0으로 초기화
for i in range(num_samples):
    amount = random_outstanding_amounts[i] + random_outstanding_counts[i] + random_delinquency_rates[i]
    for j in range(len(loan_amounts_range) - 1):
        if amount >= loan_amounts_range[j] * 10000000 and amount < loan_amounts_range[j + 1] * 10000000:
            y[i] = j
            break
    else:
        y[i] = len(loan_amounts_range) - 1

# 데이터셋 구성
X = np.column_stack((random_outstanding_amounts, random_outstanding_counts, random_delinquency_rates))

# 원-핫 인코딩
one_hot_labels = np.eye(num_classes + 1)[y]

# 모델 구성
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(3,)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes + 1, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(X, one_hot_labels, epochs=50, batch_size=32)

# 모델 저장
model.save("loan_model.h5")

# 추론을 위해 모델 로드
loaded_model = load_model("loan_model.h5")

# 예측할 데이터
new_data = np.array([[2000000, 3, 0.1]])  # 예시로 새로운 데이터를 입력합니다.

# 추론 수행
prediction = loaded_model.predict(new_data)

# 추론 결과 출력
predicted_class = np.argmax(prediction)
loan_amount = loan_amounts_range[predicted_class] * 10000000
print("Predicted loan amount:", loan_amount)
