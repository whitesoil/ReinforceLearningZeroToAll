# lab2.py 와 다른 방식으로 키 입력을 받은 코드
import gym
import readchar

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

env = gym.make('FrozenLake-v0')
env.render()
state = env.reset()

while True :
	# lab2.py 와 다른 방식으로 키 입력을 받음
	key = readchar.readkey()
	if key not in arrow_keys.keys():
		print("Game aborted")
		break

	action = arrow_keys[key]
	state, reward, done, info = env.step(action)
	env.render()

	print("S: ", state, "Action: ", action, "Reward: ", reward, "Info: ",info)

	if done:
		print("Finished with reward", reward)
		break