import numpy as np
from typing import List, Dict, Union
from collections import defaultdict

from rlkit.data_management.replay_buffer.unified_buffer import UnifiedReplayBuffer
from rlkit.data_management.replay_buffer.env_buffer import EnvReplayBuffer
from rlkit.env_creators.base_env import BaseEnv


class DPOUnifiedReplayBuffer(UnifiedReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size: int,
        env: BaseEnv,
        policy_mapping_dict: Union[Dict[str, str], None] = None,
        random_seed: int = 1995,
    ):
        self._observation_space_n = env.observation_space_n
        self._action_space_n = env.action_space_n
        self.n_agents = env.n_agents
        self.agent_ids = env.agent_ids

        if policy_mapping_dict is None:
            policy_mapping_dict = dict(
                zip(env.agent_ids, ["policy_0" for _ in range(env.n_agents)])
            )

        self._policy_buffers = defaultdict(dict)
        self._agent_buffers = {}
        for a_id, p_id in policy_mapping_dict.items():
            self._agent_buffers[a_id] = DPOEnvReplayBuffer(
                env.ego_obs_idx_n[a_id],
                max_replay_buffer_size,
                self._observation_space_n[a_id],
                self._action_space_n[a_id],
            )
            self._policy_buffers[p_id][a_id] = self._agent_buffers[a_id]

        self._max_replay_buffer_size = max_replay_buffer_size

    def add_sample(
        self,
        observation_n,
        action_n,
        reward_n,
        terminal_n,
        next_observation_n,
        pred_observation_n,
    ):
        for a_id in observation_n.keys():
            self.agent_buffers[a_id].add_sample(
                observation_n[a_id],
                action_n[a_id],
                reward_n[a_id],
                terminal_n[a_id],
                next_observation_n[a_id],
                pred_observation_n[a_id],
            )

    def get_all(self, agent_id: str, keys: List[str] = None, **kwargs):
        return self.agent_buffers[agent_id].get_all(keys=keys, **kwargs)


class DPOEnvReplayBuffer(EnvReplayBuffer):
    def __init__(self, ego_obs_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._ego_obs_idx = ego_obs_idx
        self._pred_observations = np.zeros(
            (self._max_replay_buffer_size, len(ego_obs_idx))
        )

    def clear(self):
        super().clear()
        self._pred_observations = np.zeros(
            (self._max_replay_buffer_size, len(self._ego_obs_idx))
        )

    def add_path(self, path):
        for (ob, action, reward, next_ob, pred_ob, terminal) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["pred_observations"],
            path["terminals"],
        ):
            self.add_sample(
                observation=ob,
                action=action,
                reward=reward,
                next_observation=next_ob,
                pred_observation=pred_ob,
                terminal=terminal,
            )

        self.terminate_episode()
        self._trajs += 1

    def add_sample(
        self,
        observation,
        action,
        reward,
        terminal,
        next_observation,
        pred_observation,
    ):
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._pred_observations[self._top] = pred_observation

        if terminal:
            next_start = (self._top + 1) % self._max_replay_buffer_size
            self._traj_endpoints[self._cur_start] = next_start
            self._cur_start = next_start

        if isinstance(self._observations, dict):
            for key, obs in observation.items():
                self._observations[key][self._top] = obs
            for key, obs in next_observation.items():
                self._next_obs[key][self._top] = obs
        else:
            self._observations[self._top] = observation
            self._next_obs[self._top] = next_observation
        self._advance()

    def get_all(self, keys=None, **kwargs):
        indices = range(self._size)

        return self._get_batch_using_indices(indices, keys=keys, **kwargs)

    def _get_batch_using_indices(self, indices, keys=None):
        if keys is None:
            keys = set(
                [
                    "observations",
                    "actions",
                    "rewards",
                    "terminals",
                    "next_observations",
                    "pred_observations",
                ]
            )
        if isinstance(self._observations, dict):
            obs_to_return = {}
            next_obs_to_return = {}
            for k in self._observations:
                if "observations" in keys:
                    obs_to_return[k] = self._observations[k][indices]
                if "next_observations" in keys:
                    next_obs_to_return[k] = self._next_obs[k][indices]
        else:
            obs_to_return = self._observations[indices]
            next_obs_to_return = self._next_obs[indices]

        ret_dict = {}
        if "observations" in keys:
            ret_dict["observations"] = obs_to_return
        if "actions" in keys:
            ret_dict["actions"] = self._actions[indices]
        if "rewards" in keys:
            ret_dict["rewards"] = self._rewards[indices]
        if "terminals" in keys:
            ret_dict["terminals"] = self._terminals[indices]
        if "next_observations" in keys:
            ret_dict["next_observations"] = next_obs_to_return
        if "pred_observations" in keys:
            ret_dict["pred_observations"] = self._pred_observations[indices]

        return ret_dict
