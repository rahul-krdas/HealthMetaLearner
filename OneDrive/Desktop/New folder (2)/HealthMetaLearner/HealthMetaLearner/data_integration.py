# health_meta_learner/data_integration.py
import torch
import torch.nn as nn
import torch.optim as optim

class DataIntegration:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.integration_model = IntegrationModel(modality_dims=config['modality_dims'],
                                                  hidden_dim=config['hidden_dim']).to(self.device)
        self.optimizer = optim.Adam(self.integration_model.parameters(), lr=config.get('learning_rate', 0.001))
        self.criterion = nn.MSELoss()

    def integrate_data(self, modalities_data):
        """
        Integrate multimodal healthcare data using a neural network.

        Parameters:
        - modalities_data: Dictionary containing data for different modalities.

        Returns:
        - Integrated data.
        """
        # Convert modalities data to PyTorch tensors
        modalities_tensors = {modality: torch.tensor(data, dtype=torch.float32).to(self.device)
                              for modality, data in modalities_data.items()}

        # Training the integration model
        self.train_integration_model(modalities_tensors)

        # Applying integration model to modalities data
        integrated_data = self.integration_model(modalities_tensors)

        return integrated_data.cpu().detach().numpy()

    def train_integration_model(self, modalities_tensors):
        """
        Train the integration model using data from different modalities.

        Parameters:
        - modalities_tensors: Dictionary containing tensors for different modalities.
        """
        self.integration_model.train()
        self.optimizer.zero_grad()

        integrated_data = self.integration_model(modalities_tensors)
        loss = self.criterion(integrated_data, modalities_tensors['reference_modality'])
        loss.backward()
        self.optimizer.step()

class IntegrationModel(nn.Module):
    def __init__(self, modality_dims, hidden_dim):
        super(IntegrationModel, self).__init__()
        self.fc1 = nn.Linear(in_features=sum(modality_dims.values()), out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, modalities_tensors):
        # Concatenate tensors from different modalities along the last dimension
        concatenated_data = torch.cat(list(modalities_tensors.values()), dim=-1)
        x = self.fc1(concatenated_data)
        x = self.relu(x)
        integrated_data = self.fc2(x)
        return integrated_data
