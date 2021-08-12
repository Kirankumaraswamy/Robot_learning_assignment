from __future__ import print_function

import sys

import torch

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, history_length, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events()

    # Save history
    image_hist = []
    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(-1, history_length + 1, 96, 96)

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...

        if step < 10:
            action_id = ACCELERATE
        else:

            #model_state = np.array([history_states[i] for i in range(len(history_states))])
            #model_state = np.append(model_state, np.expand_dims(state, axis=0), 0)

            #model_state = np.expand_dims(model_state, axis=0)

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            state = torch.Tensor(state).to(device)
            #model_state = model_state.view((-1, 1+history_length, 96, 96))
            action_id = agent.predict(state)
            action_id = torch.max(action_id.data, 1)[1].cpu()
            action_id = action_id.item()

        action = id_to_action(action_id, max_speed=1)
        next_state, r, done, info = env.step(action)
        episode_reward += r

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        state = np.array(image_hist).reshape(-1, history_length + 1, 96, 96)
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                # number of episodes to test
    history_length = 3

    # TODO: load agent
    agent = BCAgent(input_shape=(1, 1, 96, 96), history_length=history_length)
    agent.load("./models/bc_agent_history3.pt")
    agent.net.eval()

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_length, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    print("Mean reward: ", np.array(episode_rewards).mean())
    print("Standard Deviation: ", np.array(episode_rewards).std())
    fname = "results/results_bc_agent_history3-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
