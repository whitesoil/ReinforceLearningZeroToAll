# Non-deterministic world

import gym
import numpy as np
import matplotlib.pyplot as plt

# FrozenLake-v0 는 기본적으로 slippery가 On 되어있음, 게임 환경 생성
env = gym.make('FrozenLake-v0')

# 게임 정보를 저장할 Q 테이블 생성 16x4
Q = np.zeros([env.observation_space.n,env.action_space.n])

# Hyperparamters
learning_rate = .85 # 학습률
dis = .99 # Discount factor
num_episodes = 2000

# 성공 여부들을 저장(몇번 성공했는지 체크하기 위함)
rList = []

for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # Random Noise 
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)/(i+1))

        """
            Action에 따른 결과값 반환
            
            new_state : 다음 위치
            reward : 보상
            done : 게임이 끝났는지 여부            
	    """
        new_state, reward, done, _ = env.step(action)
        
        # Non-deterministic and Discounted reward Q-learning, update table
        Q[state,action] = (1-learning_rate) * Q[state,action] \
            + learning_rate*(reward+dis * np.max(Q[new_state,:]))

        # rAll이 0보다 크면 성공
        rAll += reward
        state = new_state

    # 성공여부를 리스트에 저장
    rList.append(rAll)

print("Score over time: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)

# 학습 결과를 그래프로 출력
plt.bar(range(len(rList)),rList,color="blue")
plt.show()
