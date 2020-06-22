import os
import gym
import numpy as np
from paddle import fluid
import parl
from parl import layers
from parl.utils import logger

LEARNING_RATE=1e-3
GAMMA=1.0

class Model(parl.Model):
    def __init__(self, act_dim):
        hd1_size=act_dim*10

        self.fc1=layers.fc(size=hd1_size, act="tanh")
        self.fc2=layers.fc(size=act_dim, act="softmax")
    
    def forward(self, obs):
        h=self.fc1(obs)
        out=self.fc2(h)
        return out

class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr):
        self.model=model
        assert isinstance(lr, float)
        self.lr=lr
    
    def predict(self, obs):
        return self.model(obs)
    
    def learn(self, obs, action, reward):
        act_prob=self.model(obs)
        #print("action is", action)
        #log_prob=layers.cross_entropy(act_prob, action)
        log_prob= layers.reduce_sum(
            -1.0*layers.log(act_prob)*layers.one_hot(
                action, act_prob.shape[1]
            ), dim=1
        )
        cost=log_prob*reward

        cost=layers.reduce_mean(cost)
        optimizer=fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim):
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        super(Agent, self).__init__(algorithm)
    
    def build_program(self):
        self.pred_program=fluid.Program()
        self.learn_program=fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs=layers.data(
                name="obs", shape=[self.obs_dim], dtype="float32"
            )
            self.act_prob=self.alg.predict(obs)
        
        with fluid.program_guard(self.learn_program):
            obs=layers.data(
                name="obs", shape=[self.obs_dim], dtype="float32"
            )
            act=layers.data(
                name="act", shape=[self.act_dim], dtype="int64"
            )
            reward=layers.data(
                name="reward", shape=[], dtype="float32"
            )
            self.cost=self.alg.learn(obs, act, reward)

    def _predict(self, obs):
        obs= np.expand_dims(obs, axis=0)
        act_prob=self.fluid_executor.run(
            self.pred_program,
            feed={"obs": obs.astype("float32")},
            fetch_list=[self.act_prob]
        )[0]
        act_prob=np.squeeze(act_prob, axis=0)
        return act_prob

    def sample(self, obs):
        act_prob=self._predict(obs)
        act=np.random.choice(range(self.act_dim), p=act_prob)
        return act

    def predict(self, obs):
        act_prob=self._predict(obs)
        act=np.argmax(act_prob)
        return act
    
    def learn(self, obs, act, reward):
        act=np.expand_dims(act, axis=-1)
        #print("act is", act)
        feed={
            "obs": obs.astype("float32"),
            "act": act.astype("int64"),
            "reward": reward.astype("float32")
        }

        cost=self.fluid_executor.run(
            self.learn_program,
            feed=feed,
            fetch_list=[self.cost]
        )[0]
        return cost

def train(env, agent):
    obs_list, action_list, reward_list=[], [], []
    obs=env.reset()
    while True:
        obs_list.append(obs)
        action=agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info= env.step(action)
        reward_list.append(reward)
        
        if done:
            break
    return obs_list, action_list, reward_list

def evaluate(env, agent, render=False):
    eval_reward=[]
    for i in range(5):
        obs=env.reset()
        episode_reward=0
        while True:
            action= agent.predict(obs)
            obs, reward, done, info= env.step(action)
            episode_reward+=reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)

    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list)-2, -1, -1):
        reward_list[i] += gamma*reward_list[i+1]
    
    return np.array(reward_list)

env=gym.make("CartPole-v0")
obs_dim=env.observation_space.shape[0]
act_dim=env.action_space.n
logger.info("obs_dim:{}, act_dim:{}".format(obs_dim, act_dim))

model=Model(act_dim=act_dim)
alg=PolicyGradient(model, lr=LEARNING_RATE)
agent=Agent(alg, obs_dim, act_dim)

max_episode=1000
for i in range(max_episode):
    obs_list, action_list, reward_list= train(env, agent)

    if i%10==0:
        logger.info(
            "Episode:{}, Reward Sum:{}.".format(i, sum(reward_list))
        )
    
    batch_obs=np.array(obs_list)
    batch_action=np.array(action_list)
    batch_reward=calc_reward_to_go(reward_list, GAMMA)

    agent.learn(obs=batch_obs, act=batch_action, reward=batch_reward)
    if (i+1)%100==0:
        total_reward=evaluate(env, agent, render=False)
        logger.info("Test reward:{}".format(total_reward))
    
