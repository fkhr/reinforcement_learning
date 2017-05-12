from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

class Agent(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def next_action(self, state: int):
        pass

class QAgent(Agent):
    def __init__(self, n_state, n_action, discount_ratio=0.99, l_rate=0.85):
        self._q = np.zeros((n_state-1, n_action))
        self._d_ratio = discount_ratio
        self._n_state = n_state
        self._n_action = n_action
        self._l_rate = l_rate

    def next_action(self, state: int):
        return np.argmax(self._q[state])

    def update(self, state: int, action: int, next_state: int, reward: float):
        td = reward - self._q[state,action]
        td += 0 if (next_state+1)==self._n_state else self._d_ratio*np.max(self._q[next_state])
        self._q[state,action] = self._q[state,action] + self._l_rate*(td)

class SimpleNN(nn.Module):
    def __init__(self, input_s, emb_dim, hidden_s, out_s):
        super().__init__()
        self.embedding = nn.Embedding(input_s, emb_dim)
        self.hidden_1 = nn.Linear(emb_dim, hidden_s)
        self.out = nn.Linear(hidden_s, out_s)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.hidden_1(x))
        x = F.sigmoid(self.out(x))
        return x

class QNetworkAgent():

    def __init__(self, n_state, n_action, hidden_s, emb_dim=5, discount_ratio=0.99, l_rate=0.01):
        self._d_ratio = discount_ratio
        self._n_state = n_state
        self.snn = SimpleNN(n_state, emb_dim, hidden_s, n_action)
        # self.optimizer = optim.Adam(self.snn.parameters(), lr=l_rate)
        self.optimizer = optim.SGD(self.snn.parameters(), lr=l_rate)
        # self.optimizer = optim.RMSprop(self.snn.parameters(), lr=l_rate)
        # self.criterion = nn.MSELoss()
        self.criterion = nn.PairwiseDistance()

        for i in self.snn.parameters():
            nn.init.normal(i, 0, 0.01)


    def next_action(self, state: int):
        if type(state) is np.int64:
            state = int(state)
        q = self.snn.forward(Variable(torch.LongTensor([state])))
        _, max_index = torch.max(q, 1)
        return int(torch.squeeze(max_index).data.numpy().squeeze())


    def update(self, state, action, next_state, reward):
        now_Q = self.snn.forward(Variable(torch.LongTensor([int(state)])))
        now_Q = torch.unsqueeze(now_Q[:,action], 0)
        if (next_state+1)==self._n_state:
            loss = self.criterion(now_Q, Variable(torch.FloatTensor([[reward]])))
        else:
            next_Q = self.snn.forward(Variable(torch.LongTensor([next_state])))
            max_Q, _ = torch.max(next_Q, 1)
            target = reward + self._d_ratio * max_Q
            loss = self.criterion(now_Q, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
