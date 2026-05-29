"""
PyTorch : 딥러닝 프레임워크
torch : PyTorch의 핵심 라이브러리로, 텐서 연산과 GPU 가속을 통한 고속 연산, 자동 미분(역전파 자동 계산) 기능을 제공한다.
torch.nn : 신경망을 구성하기 위한 모듈
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

embed_dim = 20  # 각 토큰을 20차원 벡터로 표현
atten_dim = 5  # attention 연산에서 사용하는 차원


class SelfAttention(
    nn.Module
):  # nn.Module 상속 for SelfAttention를 학습 가능한 레이어로 만들기 위함
    def __init__(self, embed_dim, attendim):
        super().__init__()  # 부모 클래스 nn.Module 초기화 코드 실행

        """
        Linear하게 재배치
        = (embed_dim)개의 값에 각각 다른 가중치를 적용하여 총 (atten_dim)개의 숫자로 나타나는 것
        = (embed_dim)차원 벡터를 (atten_dim)차원 공간으로 투영 하는 것 -> 핵심 정보만 추출

        + 이 가중치는 학습을 통해 계속 업데이트 됨
        """
        # '무엇을 찾을지'에 최적화된 공간
        self.query = nn.Linear(embed_dim, atten_dim)
        # '무엇을 제공할지'에 최적화된 공간
        self.key = nn.Linear(embed_dim, atten_dim)
        # '실제로 전달할 정보'에 최적화된 공간
        self.value = nn.Linear(embed_dim, atten_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        score = torch.matmul(
            q, k.transpose(-2, -1)
        )  # 마지막 두 차원의 자리를 바꿔 행렬곱이 가능한 형태로 만듦
        # 차원이 커질수록 내적이 커지기 때문에 보정
        score = score / k.size(-1) ** 0.5  # scaling

        attention_weights = F.softmax(
            score, dim=-1
        )  # 각 토큰(행)별로 softmax 적용 -> 각 토큰이 다른 모든 토큰에 집중하는 비율의 합 = 1
        weighted_value = torch.matmul(attention_weights, v)

        return weighted_value


"""
Note. 선형 사상

f(u + v) = f(u) + f(v)   # 합 보존
f(cu)    = c·f(u)        # 상수 곱 보존

선형 사상: 한 벡터 공간으로부터 다른 벡터 공간에 대응시킬 때,합과 상수 곱을 보존하는 것
"""

"""
Note. 내적


"""

"""
Note. 분산
1. 분산 정의
각 값이 평균으로부터 얼마나 흩어져 있는지

- 계산
1) 평균 구하기
2) 편차 계산
3) 편차 제곱
4) 편차 제곱의 평균

2. 분산의 성질
분산(c * x) = c² * 분산(x)

- 예시
x = [2, 4, 6]  →  평균 = 4, 분산 = 8/3
2x = [4, 8, 12] →  평균 = 8, 분산 = 32/3 = 4 * 8/3
"""

"""
Note. scaling

1. 기본 가정
- Q, K의 각 값이 평균: 0, 분산: 1

q = [q1, q2, q3, q4, q5]   # 각각 평균0, 분산1
k = [k1, k2, k3, k4, k5]   # 각각 평균0, 분산1

2. 내적 계산
q·k = q1*k1 + q2*k2 + q3*k3 + q4*k4 + q5*k5

3. 각 항의 분산
분산(qi * ki) = 분산(qi) * 분산(ki) = 1 * 1 = 1

4. 합산하면 분산이 누적
분산(q·k) = 분산(q1*k1) + 분산(q2*k2) + ... + 분산(q5*k5)
           = 1 + 1 + 1 + 1 + 1
           = atten_dim = 5

5. 표준편차
표준편차 = √분산 = √atten_dim = √5

6. score를 √5로 나누면 분산이 1로 정규화됨
분산(score * 1/√5) = (1/√5)² * 분산(score)
                   = (1/5) * 5
                   = 1
"""
