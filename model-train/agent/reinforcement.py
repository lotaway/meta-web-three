import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReinforcementAgent:
    
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500
    steps_done = 0
    target_update = 10

    def select_action(self, state):
        global steps_done
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * steps_done / self.epsilon_decay)
        steps_done += 1
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.policy_net(torch.FloatTensor(state))
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        num_episodes = 300

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                self.optimize_model()
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Episode {episode} Total Reward: {total_reward}")
