"""
Strategic Logic: Reinforcement Learning Agent for Beat the Streak.

Manages Double Down and Streak Saver usage using a DQN agent.

State space:
  - streak_length (0–57+)
  - p_hit (current best pick probability)
  - double_downs_remaining (0–N)
  - streak_savers_remaining (0–N)
  - phase_encoding (0=aggressive, 1=opportunistic, 2=ultra_conservative)

Actions:
  0 = pick_only       (single pick, no special action)
  1 = double_down     (use Double Down on top pick)
  2 = use_saver       (use Streak Saver if today's pick fails)
  3 = double_down_and_saver  (both, highest-risk mitigation)
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import LOG_LEVEL, MODEL_CHECKPOINT_DIR

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# ── Environment ────────────────────────────────────────────────────────────────

@dataclass
class StreakState:
    streak_length: int         = 0
    p_hit: float               = 0.75
    double_downs_remaining: int = 5
    streak_savers_remaining: int = 2
    games_played: int          = 0

    def to_array(self) -> np.ndarray:
        phase = (
            0 if self.streak_length <= 10
            else 1 if self.streak_length <= 40
            else 2
        )
        return np.array([
            self.streak_length / 57.0,        # normalized
            self.p_hit,
            self.double_downs_remaining / 5.0,
            self.streak_savers_remaining / 2.0,
            phase / 2.0,
        ], dtype=np.float32)


STATE_DIM  = 5
ACTION_DIM = 4    # pick_only, double_down, use_saver, dd_and_saver

# ── Replay Buffer ──────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Network ────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── DQN Agent ──────────────────────────────────────────────────────────────────

class StreakDQNAgent:
    """
    DQN agent for managing Beat the Streak double-downs and savers.

    Reward structure:
      - +1.0  per game with a hit
      - +10.0 bonus at milestone streaks (10, 20, 30, 40, 50, 57)
      - -20.0 if streak breaks (terminal)
      - -0.5  if used DD on a low-prob pick (poor DD usage)
    """

    MILESTONES = {10, 20, 30, 40, 50, 57}

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer     = ReplayBuffer()
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update_freq = target_update_freq
        self._step = 0

    def select_action(self, state: StreakState) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)

        state_t = torch.tensor(state.to_array(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax(dim=-1).item()

    def _compute_reward(
        self,
        action: int,
        hit_occurred: bool,
        streak_before: int,
        streak_after: int,
        p_hit: float,
        dd_used: bool,
    ) -> float:
        reward = 0.0
        if hit_occurred:
            reward += 1.0
            if streak_after in self.MILESTONES:
                reward += 10.0   # Milestone bonus
        else:
            reward -= 20.0        # Streak broken
            if dd_used:
                reward -= 0.5    # Extra penalty for wasted DD

        # Penalise using DD on low-probability picks
        if dd_used and p_hit < 0.85:
            reward -= 0.5
        return reward

    def step(
        self,
        state: StreakState,
        action: int,
        hit_occurred: bool,
        next_state: StreakState,
        done: bool,
    ) -> float:
        dd_used = action in (1, 3) and state.double_downs_remaining > 0
        reward  = self._compute_reward(
            action, hit_occurred,
            state.streak_length, next_state.streak_length,
            state.p_hit, dd_used
        )
        self.buffer.push(
            state.to_array(), action, reward,
            next_state.to_array(), float(done)
        )
        self._update()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return reward

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            t.to(self.device) for t in (states, actions, rewards, next_states, dones)
        )

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1).values
            target_q   = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: Path) -> None:
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "epsilon":      self.epsilon,
        }, path)
        log.info(f"RL agent saved to {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.policy_net.eval()
        log.info(f"RL agent loaded from {path} (ε={self.epsilon:.3f})")

    def recommend_action(self, state: StreakState) -> dict:
        """
        Greedy action recommendation (no exploration) with human-readable output.

        Returns
        -------
        dict with keys: action_id, action_name, q_values
        """
        state_t = torch.tensor(state.to_array(), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        action_id = int(q_values.argmax())
        action_names = {
            0: "pick_only",
            1: "double_down",
            2: "use_saver",
            3: "double_down_and_saver",
        }
        return {
            "action_id":   action_id,
            "action_name": action_names[action_id],
            "q_values":    {action_names[i]: round(float(q_values[i]), 4) for i in range(ACTION_DIM)},
        }
