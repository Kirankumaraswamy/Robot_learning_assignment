# export DISPLAY=:0 

import sys

import torch.nn

sys.path.append("../")

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from agent.networks import CNN
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats


STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4

def action_to_id(a):
    """
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])

def run_episode(env, agent, deterministic, skip_frames=3,  do_training=True, rendering=True, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events()

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(-1, history_length + 1, 96, 96)

    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        if step < 10:
            action_id = ACCELERATE
        else:
            action_id = agent.act(state, deterministic=deterministic)

        action = id_to_action(action_id, max_speed=0.4)
        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal:
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(-1, history_length + 1, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps :
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), "car_racing", ["episode_reward", "straight", "left", "right", "accel", "brake"])

    training_rewards = []
    validation_rewards = []
    max_timesteps = 200
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        stats = run_episode(env, agent, skip_frames=3, max_timesteps=max_timesteps, deterministic=False, do_training=True)

        train_rewad = stats.episode_reward
        training_rewards.append(train_rewad)
        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward,
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        if i % eval_cycle == 0:
            valid_reward = 0
            for j in range(num_eval_episodes):
                stats = run_episode(env, agent, history_length=history_length, deterministic=True, do_training=False)
                valid_reward += stats.episode_reward

            # taking mean
            valid_reward = valid_reward / num_eval_episodes
            validation_rewards.append(valid_reward)
            print("Validation reward: ", valid_reward)

        print("episode: %s, Training: %s" % (i, train_rewad))

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_carracin.ckpt"))

        max_timesteps = min(max_timesteps + 20, 1500)
        # epsilon decay while training
        agent.epsilon = max(0.2, (1 - (i * 3) / num_episodes) * agent.epsilon_init)
        print("agent epsilon : ", agent.epsilon)
        print("max_imesteps; ", max_timesteps)
        print("====================================================")
    tensorboard.close_session()
    return training_rewards, validation_rewards

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0



if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    num_actions = 5

    env = gym.make('CarRacing-v0').unwrapped

    # TODO: Define Q network, target network and DQN agent
    Q_network = CNN(n_classes=5)
    Q_target_network = CNN(n_classes=5)
    agent = DQNAgent(Q_network, Q_target_network, num_actions, gamma=0.9, batch_size=16, epsilon=0.8, tau=0.01, lr=0.001, history_length=0, sampling_probability=[0.54, 0.16, 0.10, 0.19, 0.01])
    agent.loss_function = torch.nn.SmoothL1Loss().cuda()
    training, validation = train_online(env, agent, num_episodes=500, history_length=0, model_dir="./models_carracing")

