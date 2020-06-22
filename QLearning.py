import gym
import numpy as np
import time

# Q(st, at)= Q(st,at)+lr*(R_t+1 + gama*max_a{Q(s_t+1, a)} - Q(st,at) )
# differs from SARSA method which is online updating, Q-learning uses off-line updating

class QLearningAgent:
    def __init__(self, action_space, state_space, lr=0.1, gama=0.9):
        self.action_space=action_space
        self.state_space = state_space
        self.lr=lr
        self.gama=gama
        self.explore_rate=0.1
        self.Q=np.random.random([self.state_space, self.action_space])

    def sample(self, state):
        _p=np.random.uniform()
        if _p<self.explore_rate:
            action=np.random.choice(self.action_space)
        else:
            action=self.predict(state)
        return action

    def predict(self, state):
        action_p=self.Q[state, :]
        Q_tsa=np.max(action_p)
        actions=np.where(action_p==Q_tsa)[0]
        #print(action_p,actions, Q_tsa, np.where(action_p==Q_tsa))
        return np.random.choice(actions)
    
    def learn(self, state, action, reward, next_state, done=False):
        predict_Q=self.Q[state, action]
        if done:
            target_Q=reward
        else:
            target_Q=reward+self.gama*np.max(self.Q[next_state, :])
        
        # update Q

        self.Q[state, action]+= self.lr*(target_Q-predict_Q)


def train(env, agent, render=False):
    total_step=0
    total_reward=0
    obs=env.reset()
    action=agent.sample(obs)
    while True:
        next_obs, reward, done, _ = env.step(action)
        next_action = agent.sample(next_obs)
        agent.learn(obs, action, reward,next_obs, done)
        action=next_action
        obs=next_obs
        total_reward+=reward
        total_step+=1
        if render:
            env.render()
        if done:
            break
    return total_reward, total_step
def test(env, agent, render=True):
    total_reward=0
    obs=env.reset()
    while True:
        action= agent.predict(obs)
        print("obs:{}, action:{}".format(obs, action))
        next_obs, reward, done, info= env.step(action)
        total_reward+=reward
        obs=next_obs
        if render:
            time.sleep(0.5)
            env.render()

        if done:
            break

    return total_reward

env=gym.make("CliffWalking-v0")
# action 0,1,2,3 up right down left

agent=QLearningAgent(env.action_space.n, env.observation_space.n  )

for episode in range(500):
    ep_reward, ep_steps = train(env, agent)
    print("episodes:{}, rewards:{}, steps:{}".format(episode, ep_reward,ep_steps))

test_reward=test(env, agent)
print("test reward:", test_reward)