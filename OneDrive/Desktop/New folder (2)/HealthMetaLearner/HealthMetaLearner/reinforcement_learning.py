# health_meta_learner/reinforcementlearning.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReinforcementLearning:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn_model = DQNModel(state_dim=config['state_dim'], action_dim=config['action_dim']).to(self.device)
        self.optimizer = optim.Adam(self.dqn_model.parameters(), lr=config.get('learning_rate', 0.001))
        self.criterion = nn.MSELoss()

    def apply_reinforcement_learning(self, state, action, reward, next_state, done):
        """
        Apply reinforcement learning for continuous improvement and adaptation.

        Parameters:
        - state: The current state.
        - action: The action taken in the current state.
        - reward: The reward received for the action.
        - next_state: The next state after taking the action.
        - done: Whether the episode is done.

        Returns:
        - None
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        # Training the DQN model
        self.train_dqn_model(state, action, reward, next_state, done)

    def train_dqn_model(self, state, action, reward, next_state, done):
        """
        Train the DQN model using experience replay.

        Parameters:
        - state: The current state.
        - action: The action taken in the current state.
        - reward: The reward received for the action.
        - next_state: The next state after taking the action.
        - done: Whether the episode is done.

        Returns:
        - None
        """
        self.dqn_model.train()
        self.optimizer.zero_grad()

        q_values = self.dqn_model(state)
        target_q_values = q_values.clone()

        with torch.no_grad():
            next_q_values = self.dqn_model(next_state)
            target_q_values[0, action] = reward + self.config['gamma'] * torch.max(next_q_values) * (1.0 - done)

        loss = self.criterion(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

class DQNModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(in_features=state_dim, out_features=64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=64, out_features=action_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        q_values = self.fc2(x)
        return q_values

