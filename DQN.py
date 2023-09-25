import torch
import torch.nn as nn
import torch.optim as optim
import numpy

import numpy as np
import copy
import random

np.random.seed(123)
random.seed(123)

GAMMA = 0.8
EPSILON = 0.1
MOVEMENT = [
    np.array([-1,0]),
    np.array([1,0]),
    np.array([0,-1]),
    np.array([0,1])
]
ACTIONS = ['up', 'down', 'left', 'right']
EPOCH_NUM = 1000

class QModel(nn.Module):
    def __init__(self, in_d, out_d, hidden_d):
        super(QModel, self).__init__()
        self.fc_1 = nn.Linear(in_d, hidden_d)
        self.fc_2 = nn.Linear(hidden_d, out_d)
        self.motivation = nn.ReLU()
        self.logit = nn.Softmax()
    def forward(self, state):
        x = self.motivation(self.fc_1(state))
        action_prob = self.logit(self.fc_2(x))
        return action_prob


class Agent(object):
    def __init__(self):
        self.start = [0,0]
        self.end = [4,4]
        self.epoch_num = EPOCH_NUM
        self.trap_pos = [[0,2], [2,2], [3,2], [4,2]]
        self.state_dim = 25
        self.action_dim = 4
        self.hidden_dim = 16
        self.movement_net = QModel(self.state_dim, self.action_dim, self.hidden_dim)
        self.target_net = QModel(self.state_dim, self.action_dim, self.hidden_dim)
        self.replay_buffer = []
        self.max_num_once = 50
        self.max_num_buffer = 1000
        self.renew_interval = 20
        self.bs = 32
        self.optimizer = optim.Adam(self.movement_net.parameters(), lr=2e-4)

    def move(self, pos, a):
        next_pos = self.trans(pos, MOVEMENT[a])
        next_pos[0] = int(next_pos[0])
        next_pos[1] = int(next_pos[1])
        reward = -1 + (next_pos == [4, 4]) * 100 - 100 * (next_pos in self.trap_pos)  
        is_done = False
        if next_pos == [4, 4] or next_pos in self.trap_pos:
            is_done = True
        return reward, next_pos, is_done
    
    def trans(self, pos, m):
        next_pos =  np.array(pos) + m 
        next_pos[0] = int(np.max([0, next_pos[0]]))
        next_pos[1] = int(np.max([0, next_pos[1]]))
        next_pos[0] = int(np.min([4, next_pos[0]]))
        next_pos[1] = int(np.min([4, next_pos[1]]))
        return list(next_pos)
    
    def state_dimension_convertor(self, state, is_2d):
        if is_2d:
            new_state = torch.zeros(25)
            new_state[int(5*state[0]+state[1])] = 1
        else:
            new_state = np.zeros(2)
            new_state[0] = int(np.where(state==1)[0]/5)
            new_state[1] = int(np.where(state==1)[0] - new_state[0]*5)
            new_state = list(new_state)
        return new_state

    
    def experience_store_once(self, state_2d):
        state_1d = self.state_dimension_convertor(state_2d, True)
        p = random.random()
        if p < EPSILON:
            action = random.randint(0,3)
        else:
            action = torch.argmax(self.movement_net(state_1d))
        state_2d = self.state_dimension_convertor(state_1d, False)
        r ,state_next_2d, is_done = self.move(state_2d, action)
        self.replay_buffer.append([state_2d, action, r, state_next_2d, is_done])
        if len(self.replay_buffer) > self.max_num_buffer:
            self.replay_buffer = self.replay_buffer[len(self.replay_buffer)-self.max_num_buffer:]
        return state_2d, action, r, state_next_2d, is_done
        
    def experience_replay(self):
        if self.bs <= len(self.replay_buffer):
            return random.sample(self.replay_buffer, self.bs)
        return self.replay_buffer
    
    def choose_a(self, pos):
        p = np.random.random()
        if p < EPSILON:
            return np.random.randint(4)
        pos_1d = self.state_dimension_convertor(pos, False)
        qvalue = self.movement_net(pos_1d)
        return torch.argmax(qvalue)


    def train(self):
        for epoch_id in range(self.epoch_num):
            state_2d = [4,0]
            loss_recorder = []
            for ts in range(self.max_num_once):
                _, action, r, state_next_2d, is_done = self.experience_store_once(state_2d)
                batch = self.experience_replay()
                loss = 0
                for sample in batch:
                    s_2d, a, r, s_n_2d, i_d = sample
                    s_1d = self.state_dimension_convertor(s_2d, True)
                    if i_d:
                        y_j = r
                    else:
                        y_j = r + GAMMA * torch.max(self.target_net(s_1d))
                    loss += (y_j - self.movement_net(s_1d)[a])**2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if ts % self.renew_interval == 0:
                    self.target_net.load_state_dict(self.movement_net.state_dict())
                state_2d = state_next_2d
                loss_recorder.append(loss.detach().numpy())
            print(epoch_id, np.mean(np.array(loss_recorder)))

    
    def show_strategy(self):
        move_matrix = np.zeros((5,5))
        for i in range(5):
            for j in range(5):
                state_1d = self.state_dimension_convertor([i, j], True)
                action = torch.argmax(self.movement_net(state_1d))
                move_matrix[i][j] = action
        print(move_matrix)
        


xiaoming = Agent()
xiaoming.train()
xiaoming.show_strategy()