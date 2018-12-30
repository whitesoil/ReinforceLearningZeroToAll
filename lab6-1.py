# Q-network for FrozenLake

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Input date를 One-hot 인코딩으로 변환
def one_hot(x):
    return np.identity(16)[x:x+1]

# FrozenLake-v0 는 기본적으로 slippery가 On 되어있음, 게임 환경 생성
env = gym.make('FrozenLake-v0')

input_size = env.observation_space.n # 16
output_size = env.action_space.n # 4
learning_rate = 0.1

X = tf.placeholder(shape=[1,input_size],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01))

Qpred = tf.matmul(X,W)
Y = tf.placeholder(shape=[1,output_size],dtype=tf.float32)

# Sum of squared error
loss = tf.reduce_sum(tf.square(Y-Qpred))

# Gradient Descent Algorithm
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99 # Discount factor
num_episodes = 2000 # 반복 횟수

rList = []

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i/50)+10) # Decaying E-greedy
        rAll = 0
        done = False
        local_loss = [] # 에러율 저장

        while not done:
            Qs = sess.run(Qpred, feed_dict = {X : one_hot(s)})

            # Decaying E-greedy
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)
            
            """
                Action에 따른 결과값 반환
                
                s1 : 다음 위치
                reward : 보상
                done : 게임이 끝났는지 여부            
	        """
            s1, reward, done, _ = env.step(a)
            if done:
                Qs[0,a] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict = {X:one_hot(s1)})
                # Discounted reward and update Q table
                Qs[0,a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X:one_hot(s),Y:Qs})

            # rAll이 0보다 크면 성공
            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: "+str(sum(rList)/num_episodes)+"%")
plt.bar(range(len(rList)),rList,color="blue")
plt.show()