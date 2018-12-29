# Non-deterministic world

import gym
import readchar

# Keyborad 입력을 Gym 코드와 바인딩
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
	'\x1b[A' : UP,
	'\x1b[B' : DOWN,
	'\x1b[C' : RIGHT,
	'\x1b[D' : LEFT
    }

# FrozenLake-v0 는 기본적으로 slippery가 On 되어있음, 게임 환경 생성
env = gym.make('FrozenLake-v0')

# 게임을 Graphic으로 렌더링
env.render()

# 현재 위치 초기화
state = env.reset()

while True :
	# lab2.py 와 다른 방식으로 키 입력을 받음
	key = readchar.readkey()
	if key not in arrow_keys.keys():
		print("Game aborted")
		break

	"""
		Action에 따른 결과값 반환
		
		state : 현재 위치
		reward : 보상
		done : 게임이 끝났는지 여부
		Info : 게임 정보
	"""
	action = arrow_keys[key]
	state, reward, done, info = env.step(action)
	env.render()

	print("S: ", state, "Action: ", action, "Reward: ", reward, "Info: ",info)

	if done:
		print("Finished with reward", reward)
		break