import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import gym
from parl.utils import logger
from replay_buffer import ReplayBuffer

LEARN_FREQ=5
MEMORY_SIZE=20000
MEMORY_WARMUP_SIZE=200
BATCH_SIZE=32
LEARNING_RATE=0.001
GAMMA=0.99

# parl framework defines Model, Algorithm and Agent

class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size=256
        hid2_size=256

        self.fc1=layers.fc(size=hid1_size, act="relu")
        self.fc2=layers.fc(size=hid2_size, act="relu")
        self.fc3=layers.fc(size=act_dim, act=None)
    
    def value(self, obs):
        h1=self.fc1(obs)
        h2=self.fc2(h1)
        Q=self.fc3(h2)
        return Q

#from parl.algorithms import DQN

class DQN(parl.Algorithm):
    def __init__(self, model, act_dim, gamma, lr):
        self.model=model
        self.target_model=copy.deepcopy(model)
        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim=act_dim
        self.gamma=gamma
        self.lr=lr
    
    def predict(self, obs):
        return self.model.value(obs)
    
    def learn(self, obs, action, reward, next_obs, terminal):
        next_pred_value=self.target_model.value(next_obs)
        best_v=layers.reduce_max(next_pred_value, dim=-1)
        best_v.stop_gradient=True
        terminal=layers.cast(terminal, dtype="float32")
        target= reward+ (1.0-terminal)*self.gamma*best_v

        pred_value=self.model.value(obs)
        action_onehot=layers.one_hot(action, self.act_dim)
        action_onehot=layers.cast(action_onehot, dtype="float32")
        pred_action_value=layers.reduce_sum(
            layers.elementwise_mul(pred_value, action_onehot), dim=-1
        )

        cost=layers.square_error_cost(target, pred_action_value)
        cost=layers.reduce_mean(cost)
        optimizer=fluid.optimizer.Adam(learning_rate=self.lr)
        optimizer.minimize(cost)
        return cost

    def async_target(self):
        self.model.sync_weights_to(self.target_model)

class Agent(parl.Agent):
    def __init__(self, algorithm, obs_dim, act_dim, e_greedy=0.1, 
        e_greedy_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim=obs_dim
        self.act_dim=act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step=0
        self.update_target_steps=200

        self.e_greedy=e_greedy
        self.e_greedy_decrement=e_greedy_decrement

    def build_program(self):
        self.pred_program=fluid.Program()
        self.learn_program=fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs=layers.data(
                name="obs", shape=[self.obs_dim], dtype="float32"
            )
            self.value=self.alg.predict(obs)


        with fluid.program_guard(self.learn_program):
            obs=layers.data(
                name="obs", shape=[self.obs_dim], dtype="float32"
            )
            action=layers.data(
                name="act", shape=[1], dtype="int32"
            )
            reward=layers.data(
                name="reward", shape=[], dtype="float32"
            )
            next_obs=layers.data(
                name="next_obs", shape=[self.obs_dim], dtype="float32"
            )
            terminal=layers.data(
                name="terminal", shape=[], dtype="bool"
            )

            self.cost=self.alg.learn(
                obs, action, reward, next_obs, terminal
            )
    
    def sample(self, obs):
        sample=np.random.rand()
        if sample< self.e_greedy:
            act=np.random.randint(0, self.act_dim)
        else:
            act=self.predict(obs)
        self.e_greedy=max(
            0.01, self.e_greedy-self.e_greedy_decrement
        )

        return act

    def predict(self, obs):
        obs=np.expand_dims(obs, axis=0)
        pred_Q=self.fluid_executor.run(
            self.pred_program,
            feed={"obs":obs.astype("float32")},
            fetch_list=[self.value]
        )[0]
        pred_Q=np.squeeze(pred_Q, axis=0)
        act=np.argmax(pred_Q)

        return act
    
    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step%self.update_target_steps==0:
            self.alg.async_target()
        self.global_step+=1

        act=np.expand_dims(act, -1)

        feed={
            "obs": obs.astype("float32"),
            "act": act.astype("int32"),
            "reward": reward.astype("float32"),
            "next_obs": next_obs.astype("float32"),
            "terminal": terminal
        }
        cost=self.fluid_executor.run(
            self.learn_program, feed=feed, 
            fetch_list=[self.cost]
        )[0]

        return cost
    
def train(env, agent, rpm):
    total_reward=0
    obs= env.reset()
    step=0

    while True:
        step+=1
        action=agent.sample(obs)
        next_obs, reward, done, _= env.step(action)
        rpm.append([obs, action, reward, next_obs, done])

        if len(rpm)> MEMORY_WARMUP_SIZE and step% LEARN_FREQ==0:
            (batch_obs, batch_action, batch_reward, 
                batch_next_obs, batch_done)=rpm.sample(BATCH_SIZE)
            train_loss=agent.learn(
                batch_obs, batch_action, batch_reward,
                batch_next_obs, batch_done
            )

        total_reward+=reward
        obs=next_obs
        if done:
            break
    
    return total_reward


def evaluate(env, agent, render=False):
    eval_reward=[]
    for i in range(5):
        obs= env.reset()
        episode_reward=0
        while True:
            action= agent.predict(obs)
            obs, reward, done, _ =env.step(action)
            episode_reward+=reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)

    return np.mean(eval_reward)


#env=gym.make("CartPole-v0")
env=gym.make('MountainCar-v0')
action_dim=env.action_space.n
obs_shape=env.observation_space.shape

rpm=ReplayBuffer(MEMORY_SIZE)

model=Model(act_dim=action_dim)
algorithm=DQN(model, action_dim, GAMMA, LEARNING_RATE)
agent=Agent(algorithm, obs_shape[0], action_dim, 0.5, 1e-6)

while len(rpm)< MEMORY_WARMUP_SIZE:
    train(env, agent, rpm)

max_episode= 3000

episode=0

while episode< max_episode:
    for i in range(50):
        total_reward=train(env, agent, rpm)
        episode+=1

    eval_reward=evaluate(env, agent, True)
    logger.info(
        "episodes:{}, e_greedy:{}, test_reward:{}".format(
            episode, agent.e_greedy, eval_reward
        )
    )

eval_reward=evaluate(env, agent, True)
logger.info("*"*20+"\nreward={}".format(eval_reward))
