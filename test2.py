import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 랜덤 데이터 생성
np.random.seed(42)  # 랜덤 시드 설정
num_samples = 1000  # 데이터 샘플 개수

outstanding_amounts = np.random.randint(1000000, 5000000, num_samples)  # 미상환금액
outstanding_counts = np.random.randint(1, 10, num_samples)  # 미상환건수
delinquency_rates = np.random.uniform(0.0, 1.0, num_samples)  # 연체율
loan_amounts = np.random.randint(30000000, 45000001, num_samples)  # 대출금액

# 대출 금액 범위 설정
min_loan_amount = 30000000  # 최소 대출 금액
max_loan_amount = 45000000  # 최대 대출 금액
step_size = 1000000  # 대출 금액의 이동 단위

# 대출 금액 범위 생성
loan_amounts_range = list(range(min_loan_amount, max_loan_amount + step_size, step_size))

# 데이터셋 구성
X = np.column_stack((outstanding_amounts, outstanding_counts, delinquency_rates, loan_amounts))
y = []

# 대출 금액에 해당하는 레이블 배치
for i in range(len(outstanding_amounts)):
    amount = outstanding_amounts[i] + outstanding_counts[i] + delinquency_rates[i] + loan_amounts[i]
    for j in range(len(loan_amounts_range) - 1):
        if amount >= loan_amounts_range[j] and amount < loan_amounts_range[j + 1]:
            y.append(j)
            break
    else:
        y.append(len(loan_amounts_range) - 1)

# 원-핫 인코딩
num_classes = len(loan_amounts_range)
one_hot_labels = np.eye(num_classes)[y]

# 모델 구성
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
model.fit(X, one_hot_labels, epochs=50, batch_size=32)

# 모델 저장
model.save("loan_model.h5")

# 새로운 데이터로 추론
new_data = np.array([[200000000, 5, 0.8, 40000000]])  # 예시 데이터
prediction = model.predict(new_data)
loan_amount_index = np.argmax(prediction)
loan_amount = loan_amounts_range[loan_amount_index]
print("Predicted Loan Amount:", loan_amount)
