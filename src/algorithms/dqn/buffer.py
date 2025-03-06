from typing import NamedTuple, Tuple, Union
from collections import deque
import random
import numpy as np
import torch

class TorchTensor(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor

class ExperienceReplayBuffer:
    def __init__(
        self,
        buffer_size: int=10000,
        batch_size: int=64,
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def add(
        self,
        state: Union[np.ndarray, Tuple],
        action: int,
        reward: Union[int, float],
        next_state: Union[np.ndarray, Tuple],
        done: bool,
        ) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    def get(self) -> TorchTensor:
        data = random.sample(self.buffer, self.batch_size)
        
        batch_data = (
            np.stack([x[0] for x in data]).astype(np.float32), # state
            np.array([x[1] for x in data]).astype(np.int32), # action
            np.array([x[2] for x in data]).astype(np.float32), # reward
            np.stack([x[3] for x in data]).astype(np.float32), # next_state
            np.array([x[4] for x in data]).astype(np.int32), # done
        )
        return TorchTensor(*tuple(map(self.to_torch, batch_data)))
    def to_torch(self, array: np.ndarray) -> torch.Tensor:
        return torch.tensor(array, dtype=torch.float32, device=self.device)  
