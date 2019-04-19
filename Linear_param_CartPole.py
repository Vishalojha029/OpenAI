import numpy as np
import random
import sys
import gym
env = gym.make("CartPole-v1")
W=np.random.randn(1,10)
epsilon=0.15
learning_rate=0.1
gamma=0.99
def epsilon_greedy(epsilon,state,W):
    
	 if np.random.random() < epsilon:
         
		  return env.action_space.sample()
	 else:
         
		  return argmax(state,W)
    
def argmax(S,W):
	Q1=Q(make_features(S,1),W)
	Q2=Q(make_features(S,0),W)
	return 1 if Q1 >Q2 else 0

def amax(next_state,W):
    Q1=Q(make_features(next_state,1),W)
    Q2=Q(make_features(next_state,0),W)
    return Q1 if Q1>Q2 else Q2

def Q(X,W):
    X=X.reshape(10,1)
    return np.dot(W,X)
     
 
def make_features(obs, act):
    features = np.ones(5) 
    features[1:5] = obs
    zeros = np.zeros_like(features)
    if act == 0:
        return np.append(features, zeros)
    else:
        return np.append(zeros, features)


for e in range(10000):
    state = env.reset()
    for t in range(3000):
        #env.render()
        action=epsilon_greedy(epsilon,state,W)
        X=make_features(state,action)
        X=X.reshape(10,1)
        next_state,reward,done,_=env.step(action)
        W=W+learning_rate*np.dot(((reward+gamma*amax(next_state,W))-Q(X,W)),X.T)
        state=next_state
        if done :
            print("Episode {} finished after {} timesteps".format(e+1, t+1))
            break
