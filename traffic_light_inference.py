import numpy as np
from tensorflow import keras

# 샘플 데이터 생성
signals = ["빨간색", "노란색", "초록색"]
actions = ["전진", "정지"]

# 데이터셋 생성
data = []
labels = []
for _ in range(100):
    signal = np.random.choice(signals)
    if signal == "노란색":
        action = "정지"
    else:
        action = "정지" if signal == "빨간색" else "전진"
    data.append(signal)
    labels.append(action)

# 신호등 색상을 정수로 매핑
signal_mapping = {signal: i for i, signal in enumerate(signals)}
mapped_data = [signal_mapping[signal] for signal in data]

# 차량 상태를 정수로 매핑
action_mapping = {action: i for i, action in enumerate(actions)}
mapped_labels = [action_mapping[action] for action in labels]

# 데이터를 원-핫 인코딩으로 변환
signal_encoded = keras.utils.to_categorical(mapped_data, num_classes=len(signals))
action_encoded = keras.utils.to_categorical(mapped_labels, num_classes=len(actions))

# 모델 구성
model = keras.Sequential()
model.add(keras.layers.Dense(32, input_dim=len(signals), activation="relu"))
model.add(keras.layers.Dense(len(actions), activation="softmax"))

# 모델 컴파일
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# 모델 학습
model.fit(signal_encoded, action_encoded, epochs=10, batch_size=32)

# 학습된 모델을 사용하여 예측
test_signal = "노란색"  # 테스트용 신호등 색상
mapped_test_signal = signal_mapping[test_signal]
test_signal_encoded = keras.utils.to_categorical([mapped_test_signal], num_classes=len(signals))
predictions = model.predict(test_signal_encoded)
predicted_action_index = np.argmax(predictions)
predicted_action = actions[predicted_action_index]

print("테스트 신호등 색상:", test_signal)
print("예측된 차량 상태:", predicted_action)
