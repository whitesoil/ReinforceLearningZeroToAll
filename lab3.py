# Dummy Q-learning algorithm

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# argmax는 값들이 같을 때 맨 처음 것을 선택함
# rargmax로 값들이 같아도 그들 중 랜덤하게 선택
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4','is_slippery':False}
)

env = gym.make('FrozenLake-v3')

# 테이블 생성 16x4
Q = np.zeros([env.observation_space.n,env.action_space.n]) 

# 반복 횟수
num_episodes = 2000

# 성공 여부들을 저장(몇번 성공했는지 체크하기 위함)
rList = []
for i in range(num_episodes):    
    state = env.reset() # env 초기화
    rAll = 0 #성공(게임 클리어) 여부
    done = False
    while not done:
        action = rargmax(Q[state,:])
        new_state, reward, done, _ = env.step(action)

        # Q 테이블 업데이트
        Q[state, action] = reward + np.max(Q[new_state,:])

        # reward가 0보다 크다면 성공
        rAll += reward
        state = new_state

    # 성공여부를 리스트에 저장
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)),rList, color="blue")
plt.show()