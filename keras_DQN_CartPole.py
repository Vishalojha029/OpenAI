# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:00:01 2019

@author: vishal
"""

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import  gym
import os

env=gym.make("CartPole-v0")
action_space=env.action_space.n
state_space=env.observation_space.shape[0]
num_episode=10000
batch_size=32

class DQNAgent:
    def __init__(self,action_space,state_space):
        self.action_space=action_space
        self.state_space=state_space
        self.learning_rate=0.001
        self.gamma=0.95
        self.epsilon=0.1
        self.epsilon_decay=0.9995
        self.min_epsilon=0.01
        self.model=self.build_model()
        self.memory=deque(maxlen=5000)
        self.decay_lr=0.999
        
    def build_model(self):
        model=Sequential()
        model.add(Dense(24,input_dim=state_space,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(action_space,activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        return model
    def epsilon_greedy(self,S):
        if np.random.randn(1)>self.epsilon:
            return env.action_space.sample()
        action=self.model.predict(S)
        return np.argmax(action[0])
    def add_memory(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))
    def replay(self,batch_size):
        mini_batch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in mini_batch:
            target=reward
            if not done:
                target=reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            targetf=self.model.predict(state)
            targetf[0][action]=target
            self.model.fit(state,targetf,epochs=1,verbose=0)
        if self.epsilon>self.min_epsilon:
            self.epsilon*=self.epsilon_decay
           
agent=DQNAgent(action_space,state_space)
done=False
for e in range(num_episode):
    state=env.reset()
    state=np.reshape(state,[1,state_space])
    for time in range(5000):
        #env.render()
        action=agent.epsilon_greedy(state)
        next_state,reward,done,info=env.step(action)
        reward=-50 if done and time<500  else reward
        next_state=np.reshape(next_state,[1,state_space])
        agent.add_memory(state,action,reward,next_state,done)
        state=next_state
        if done:
            print("episode {} teminate after time step {}".format(e+1,time+1))
            break
        if len(agent.memory)>batch_size:
            agent.replay(batch_size)
        
    
    
    
    
    
    
    