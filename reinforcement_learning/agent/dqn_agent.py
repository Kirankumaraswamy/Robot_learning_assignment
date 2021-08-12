import tensorflow as tf
import numpy as np
from agent.replay_buffer import ReplayBuffer
from collections import namedtuple
from torch.autograd import Variable
import torch

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.1, tau=0.01, lr=1e-4,
                 history_length=0, sampling_probability=[]):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon_init = epsilon
        self.epsilon = epsilon

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions
        self.sampling_probability = sampling_probability
        self.history_length = history_length

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update:
        #       2.1 compute td targets and loss
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        if len(self.replay_buffer._data.states) < self.batch_size:
            return

        states, actions, next_states, rewards, terminals = self.replay_buffer.next_batch(self.batch_size)

        states = torch.from_numpy(states).float().cuda().squeeze(1)
        actions = torch.from_numpy(actions).long().cuda()
        next_states = torch.from_numpy(next_states).float().cuda().squeeze(1)

        target = rewards + (1 - terminals) * self.gamma * torch.max(self.Q_target(next_states), dim=1)[
            0].cpu().detach().numpy()

        current_prediction = self.Q(states)[torch.arange(self.batch_size).long(), actions]

        self.optimizer.zero_grad()
        loss = self.loss_function(current_prediction, torch.from_numpy(target).float().cuda())
        loss.backward()
        self.optimizer.step()

        soft_update(self.Q_target, self.Q, self.tau)

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            action_id = np.argmax(self.Q(tt(state)).cpu().detach().numpy())
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = np.random.choice(np.arange(0, self.num_actions), p=self.sampling_probability)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))