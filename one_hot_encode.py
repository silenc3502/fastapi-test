import numpy as np

# 신호등 상태 정의
signal_light = ['빨간색', '노란색', '초록색']

# 신호등 상태를 원-핫 인코딩
def one_hot_encode(signal):
    encoded_signal = np.zeros(len(signal_light))
    encoded_signal[signal_light.index(signal)] = 1
    return encoded_signal

# 신호등 상태 출력 및 인코딩 결과 확인
current_signal = '초록색'
encoded_signal = one_hot_encode(current_signal)
print('현재 신호등 상태:', current_signal)
print('인코딩 결과:', encoded_signal)
