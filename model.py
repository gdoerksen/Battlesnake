import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from pathlib import Path

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # this executes the prediction via feed forward
        # i.e. this is like calling model.predict()
        # x is the tensor
        x = F.relu(self.linear1(x))
        x = (self.linear2(x))
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = Path(__file__).parent / "model"
        model_folder_path.mkdir(exist_ok=True, parents=True) #TODO check if this works
        file_name = model_folder_path / file_name
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate) #TODO review optimizer selection
        self.criterion = nn.MSELoss() #TODO review loss function selection

    def train_step(self, state, action, reward, next_state, is_game_over):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            is_game_over = (is_game_over, )

        # simplified bellman equation?

        # 1. predict Q values for current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(is_game_over)):
            Q_new = reward[idx]
            if not is_game_over[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2. Q_new = r + gamma * max(next_state)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) # calculate loss function
        loss.backward() # applies backpropagation

        self.optimizer.step()

