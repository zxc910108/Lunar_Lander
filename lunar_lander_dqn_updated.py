import gym
import numpy as np
from collections import deque
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear
import random

class DQN_update:

    def __init__(self,action_space,state_space,lr,epsilon_decay,gamma):
        self.action_space=action_space
        self.state_space=state_space
        self.epsilon=1
        self.epsilon_min=0.1
        self.gamma=gamma
        self.lr=lr
        self.replay_memory= deque(maxlen=1000000)
        self.batch_size=64
        self.e_decay=epsilon_decay
        self.model=self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_counter=0

    def build_model(self):
        model=Sequential()
        model.add(Dense(150,input_dim=self.state_space,activation=relu))
        model.add(Dense(75,activation=relu))
        model.add(Dense(self.action_space,activation=linear))
        model.compile(loss='mse',optimizer=adam(lr=self.lr))
        return model

    def take_egreedy_action(self,state):
        rprob=np.random.uniform(0,1)
        if rprob <self.epsilon:
            action = np.random.randint(0,self.action_space)
        else:
            action_vals=self.model.predict(state)
            action=np.argmax(action_vals[0])
        return action

    def take_action(self,state):
        action_vals=self.model.predict(state)
        action=np.argmax(action_vals[0])
        return action

    def add_memory(self,experience):
        self.replay_memory.append(experience)

    def train_replay(self,end_episode):
        if len(self.replay_memory)<self.batch_size:
            return

        minibatch=random.sample(self.replay_memory,self.batch_size)
        current_states=np.array([experience[0] for experience in minibatch])
        current_states=np.squeeze(current_states)
        #print("the size of current_states is {}".format(current_states.shape))

        #self.model.summary()
        current_qs_batch=self.model.predict_on_batch(current_states)

        next_states=np.array([experience[3] for experience in minibatch])
        next_states=np.squeeze(next_states)
        future_qs_batch= self.target_model.predict_on_batch(next_states)

        x=current_states
        y=[]

        for index, (current_state, action,reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_future_q=np.max(future_qs_batch[index])
                #print("the maximum future q is {}".format(max_future_q))
                new_q=reward+self.gamma*max_future_q
            else:
                new_q=reward

            current_qs=current_qs_batch[index]
            current_qs[action]=new_q

            y.append(current_qs)
        if end_episode:
            self.target_counter+=1

        if self.target_counter > 20 :
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


        history=self.model.fit(x,np.array(y),verbose=0,epochs=1)
        avg_loss=sum(history.history['loss'])/self.batch_size
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.e_decay
        return avg_loss

def test_dqn(agent):
    episode_r=0
    state=env.reset()
    state = np.reshape(state, (1, 8))
    done=False
    while not done:
        action=agent.take_action(state)
        next_state,r, done,info=env.step(action)
        next_state = np.reshape(next_state, (1, 8))
        episode_r+=r
        state=next_state
    return episode_r


def train_test_dqn(agent,episodes):

    train_score=[]
    test_score=[]
    nn_loss=[]

    #agent=DQN(env.action_space.n, env.observation_space.shape[0],lr,epsilon_decay,gamma)
    for e in range(episodes):
        state=env.reset()
        state = np.reshape(state, (1, 8))
        max_steps=1000
        score=0
        episode_loss=0
        step_count=0
        for i in range(max_steps):
            action=agent.take_egreedy_action(state)
            next_state,r, done,info=env.step(action)
            next_state = np.reshape(next_state, (1, 8))
            score+=r
            agent.add_memory((state, action,r,next_state,done))
            avg_loss=agent.train_replay(done)
            if avg_loss is not None:
                episode_loss+=avg_loss
            state=next_state
            step_count+=1
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, score))
                break
        test_score.append(test_dqn(agent))
        nn_loss.append(episode_loss/step_count)
        train_score.append(score)

         # Average score of last 100 episode
        is_solved = np.mean(test_score[-100:])
        if is_solved >=200:
            print('Task Completed!')
            break
        print("Average over last 100 episode: {0:.2f} \n".format(is_solved))

    return train_score,nn_loss,test_score

if __name__=='__main__':

    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    ##########################
    ###solve lunar lander#####
    ##########################
    episodes = 500


    agent=DQN_update(env.action_space.n, env.observation_space.shape[0],lr=0.001,epsilon_decay=0.995,gamma=0.99)
    loss,nn_loss,test_score= train_test_dqn(agent=agent,episodes=episodes)

    plt.figure(figsize=(10,6))
    plt.plot([i+1 for i in range(0, len(loss), 2)], loss[::2])
    plt.ylabel('Training Episode Reward',size=20)
    plt.xlabel('Episode',size=20)
    plt.title('Episode score (training)',size=15)
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(test_score[-100:])
    plt.ylabel('Testing Episode Reward',size=20)
    plt.xlabel('Episode',size=20)
    plt.title('Episode score (test)',size=15)
    plt.show()

    print ("The average test reward of last 100 trails is {}".format(np.mean(test_score[-100:])))
