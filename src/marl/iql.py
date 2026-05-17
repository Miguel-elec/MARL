"""
iql.py
------
Independent Q-Learning (IQL):
    Cada semáforo entrena su propio DQN de forma completamente independiente.
    Es la línea base más simple; sirve para validar que el entorno funciona.

Componentes:
    ReplayBuffer  → memoria de experiencia circular
    QNetwork      → red neuronal Q(obs, a)
    IQLAgent      → un agente DQN (un semáforo)
    IQL           → coordina todos los agentes
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Red neuronal Q
# ─────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Agente individual (un semáforo)
# ─────────────────────────────────────────────────────────────────────────────

class IQLAgent:
    """DQN con target network y epsilon-greedy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float             = 1e-3,
        gamma: float          = 0.99,
        epsilon_start: float  = 1.0,
        epsilon_end: float    = 0.05,
        epsilon_decay: float  = 0.9995,
        buffer_size: int      = 50_000,
        batch_size: int       = 64,
        target_update: int    = 200,   # pasos entre actualizaciones de target
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net     = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_size)

        self._update_count = 0

    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.q_net(t).argmax(dim=1).item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs      = torch.tensor(obs,      device=self.device)
        actions  = torch.tensor(actions,  device=self.device)
        rewards  = torch.tensor(rewards,  device=self.device)
        next_obs = torch.tensor(next_obs, device=self.device)
        dones    = torch.tensor(dones,    device=self.device)

        # Q(s, a) actuales
        q_vals = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a')
        with torch.no_grad():
            next_q  = self.target_net(next_obs).max(1).values
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_vals, targets)   # Huber loss (más estable que MSE)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # Actualizar target network
        self._update_count += 1
        if self._update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay de exploración
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        torch.save({"q_net": self.q_net.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinador multi-agente
# ─────────────────────────────────────────────────────────────────────────────

class IQL:
    """
    Un IQLAgent por semáforo.
    Uso:
        iql = IQL(env.obs_dims, env.action_dims)
        actions = iql.act(obs)
        iql.store(obs, actions, rewards, next_obs, done)
        losses = iql.update()
    """

    def __init__(self, obs_dims: dict[str, int], action_dims: dict[str, int], **kwargs):
        self.agents: dict[str, IQLAgent] = {
            ts_id: IQLAgent(obs_dim, action_dims[ts_id], **kwargs)
            for ts_id, obs_dim in obs_dims.items()
        }

    def act(self, observations: dict[str, np.ndarray], greedy: bool = False) -> dict[str, int]:
        return {
            ts_id: agent.select_action(observations[ts_id], greedy=greedy)
            for ts_id, agent in self.agents.items()
        }

    def store(self, obs, actions, rewards, next_obs, done: bool):
        for ts_id, agent in self.agents.items():
            agent.store(obs[ts_id], actions[ts_id], rewards[ts_id],
                        next_obs[ts_id], float(done))

    def update(self) -> dict[str, float]:
        losses = {}
        for ts_id, agent in self.agents.items():
            loss = agent.update()
            if loss is not None:
                losses[ts_id] = loss
        return losses

    def save(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        for ts_id, agent in self.agents.items():
            agent.save(os.path.join(directory, f"{ts_id}.pt"))

    def load(self, directory: str):
        import os
        for ts_id, agent in self.agents.items():
            path = os.path.join(directory, f"{ts_id}.pt")
            if os.path.exists(path):
                agent.load(path)

    @property
    def mean_epsilon(self) -> float:
        return sum(a.epsilon for a in self.agents.values()) / len(self.agents)
