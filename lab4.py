# Q learning exploration and discounted reward algorithm

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

# 게임 환경 셋팅
register(
    id='FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery' : False}
)

# 게임 환경 생성
env = gym.make('FrozenLake-v3')

# 게임 정보를 저장할 Q 테이블 생성 16x4
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Discount factor
dis = .99

# 반복 횟수
num_episodes = 2000

# 성공 여부들을 저장(몇번 성공했는지 체크하기 위함)
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # Decaying E-greedy
    e = 1. / ((i // 100)+1)
    while not done:
        # Random Noise 
        action = np.argmax(Q[state,:]+np.random.randn(1,env.action_space.n)/(i+1))
        
        # Decaying E-greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state,:])

        """
            Action에 따른 결과값 반환
            
            new_state : 다음 위치
            reward : 보상
            done : 게임이 끝났는지 여부            
	    """
        new_state, reward, done, _ = env.step(action)

        # Discounted reward and update Q table
        Q[state,action] = reward + dis * np.max(Q[new_state,:])

        # rAll이 0보다 크면 성공
        rAll += reward
        state = new_state

    # 성공여부를 리스트에 저장
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q) 

# 학습 결과를 그래프로 출력
plt.bar(range(len(rList)),rList, color="blue")
plt.show()