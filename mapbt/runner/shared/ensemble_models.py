import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Model, self).__init__()
        self.activation = torch.relu
        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            layer = nn.Linear(last_dim, nh)
            nn.init.xavier_uniform_(layer.weight)
            self.affine_layers.append(layer)
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        action_mean = self.action_mean(x)
        return action_mean
    
class Ensemble(nn.Module):
    def __init__(self, observation_shape, action_shape, device, hidden_sizes=(256, 256), num_nets=8):
        super(Ensemble, self).__init__()
        
        obs_dim = observation_shape[0]
        act_dim = action_shape[0]
        self.device = device
        self.num_nets = num_nets

        self.pis = []
        for _ in range(num_nets):
            pi = Model(obs_dim, act_dim, hidden_sizes).to(device).float()
            self.pis.append(pi)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0:  # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.mean(np.array(vals), axis=0)

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vals = list()
            for pi in self.pis:
                vals.append(pi(obs).cpu().numpy())
            return np.square(np.std(np.array(vals), axis=0)).mean()

    def load(self, path):
        state = torch.load(path)
        for net_id, net_state in state.items():
            self.pis[int(net_id[-1])].load_state_dict(net_state)

    def save(self, path):
        state = {"ensemble_net_{}".format(i): self.pis[i].state_dict() for i in range(len(self.pis))}
        torch.save(state, path)
        
class EnsembleDAgger:
    def __init__(self, ensemble_model: Ensemble, threshold: float = 150.0):
        self.ensemble = ensemble_model
        self.threshold = threshold

    def query_expert(self, obs):
        obs_np = obs if isinstance(obs, np.ndarray) else np.array(obs)
        var = self.ensemble.variance(obs_np)
        print(var)
        return var > self.threshold
    def query_expert2(self, obs):
        obs_np = obs if isinstance(obs, np.ndarray) else np.array(obs)
        var = self.ensemble.variance(obs_np)
        return var