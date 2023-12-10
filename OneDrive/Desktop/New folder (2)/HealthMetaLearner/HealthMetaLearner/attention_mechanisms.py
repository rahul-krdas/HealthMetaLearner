# health_meta_learner/attention_mechanism.py
import torch
import torch.nn as nn
import torch.optim as optim

class AttentionMechanism:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_model = AttentionModel(input_dim=config['input_dim'], hidden_dim=config['hidden_dim']).to(self.device)
        self.optimizer = optim.Adam(self.attention_model.parameters(), lr=config.get('learning_rate', 0.001))
        self.criterion = nn.MSELoss()

    def apply_attention(self, input_data, meta_features):
        """
        Apply attention mechanisms for enhanced interpretability in healthcare recommendation.

        Parameters:
        - input_data: The input healthcare data.
        - meta_features: Meta-features obtained from meta-learning.

        Returns:
        - Output data with attention applied.
        """
        input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
        meta_features = torch.tensor(meta_features, dtype=torch.float32).to(self.device)

        # Training the attention model
        self.train_attention_model(input_data, meta_features)

        # Applying attention to input data
        attention_weights = self.attention_model(meta_features)
        output_data = input_data * attention_weights

        return output_data.cpu().detach().numpy()

    def train_attention_model(self, input_data, meta_features):
        """
        Train the attention model using meta-features and input data.

        Parameters:
        - input_data: The input healthcare data.
        - meta_features: Meta-features obtained from meta-learning.
        """
        self.attention_model.train()
        self.optimizer.zero_grad()

        attention_weights = self.attention_model(meta_features)
        predicted_data = input_data * attention_weights

        loss = self.criterion(predicted_data, input_data)
        loss.backward()
        self.optimizer.step()

class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        attention_weights = self.sigmoid(x)
        return attention_weights
