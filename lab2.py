# OpenAI Gym Tutorial

import gym
from gym.envs.registration import register
import sys, tty, termios

# Keyborad 입력 처리
class _Getch:
	def __call__(self):
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(3)
		finally:
			termios.tcsetattr(fd,termios.TCSADRAIN, old_settings)
		return ch

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
	'\x1b[A' : UP,
	'\x1b[B' : DOWN,
	'\x1b[C' : RIGHT,
	'\x1b[D' : LEFT}

# OpenAI Gym의 게임 환경 셋팅
register(
	id='FrozenLake-v3',
	entry_point='gym.envs.toy_text:FrozenLakeEnv',
	kwargs={'map_name' : '4x4', 'is_slippery': False}
)

# 게임 환경 생성
env = gym.make('FrozenLake-v3')

# 게임을 Graphic으로 렌더링
env.render()

while True:
	key = inkey()
	if key not in arrow_keys.keys():
		print("Game aborted")
		break

	action = arrow_keys[key]

	"""
		Action에 따른 결과값 반환
		
		state : 현재 위치
		reward : 보상
		done : 게임이 끝났는지 여부
		Info : 게임 정보
	"""
	state, reward, done, info = env.step(action)
	env.render()
	print("S: ", state, "Action: ", action, "Reward: ", reward, "Info: ",info)

	if done:
		print("Finished with reward", reward)
		break