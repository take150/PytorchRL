import copy
from typing import Union

import random
import numpy as np
import torch    
from torch import nn
from torch import optim

from src.algorithms.dqn.buffer import ExperienceReplayBuffer  
from src.algorithms.dqn.net import Net  

class Agent:
    def __init__(self, cfg: dict):
        self.lr = cfg['learning_rate']
        self.gamma = cfg['discount_rate']
        self.buffer_size = cfg['buffer_size']
        self.batch_size = cfg['batch_size']
        self.target_update = cfg['target_update_steps']
        
        self.state_size = cfg['state_size']
        self.action_size = cfg['action_size']

        self.original_qnet = Net(self.state_size, self.action_size)
        self.target_qnet = Net(self.state_size, self.action_size)
        self.sync_net()

        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / cfg['total_timesteps']
        self.epsilon = self.epsilon_start

        self.replay = ExperienceReplayBuffer(self.buffer_size, self.batch_size)
        self.data = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = optim.Adam(self.original_qnet.parameters(), self.lr)
    def get_action(self, state) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :].astype(np.float32), device=self.device)
            q_c = self.original_qnet(state)
            return q_c.detach().argmax().item()
    def update(self) -> None:
        if len(self.replay.buffer) < self.batch_size:
            return
        self.data = self.replay.get()
        q_c = self.original_qnet(self.data.state)
        q = q_c[np.arange(self.batch_size), self.data.action.cpu().numpy()]
        
        next_q_c = self.target_qnet(self.data.next_state)
        next_q = next_q_c.max(1)[0]
        next_q.detach()
        target = self.data.reward + (1 - self.data.done) * self.gamma *  next_q

        loss_function = nn.MSELoss()
        loss = loss_function(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        # for name, param in self.original_qnet.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad:", param.grad)
        # print(q, target)
        self.optimizer.step()
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: Union[int, float],
        next_state: np.ndarray,
        done: bool,
        ) -> None:
        self.replay.add(state, action, reward, next_state, done)
    def sync_net(self) -> None:
        self.target_qnet = copy.deepcopy(self.original_qnet)
    def set_epsilon(self) -> None:
        self.epsilon -= self.epsilon_decay
    def save_model(self) -> None:
        torch.save(self.original_qnet.state_dict(), 'model.pth')
