from typing import Dict, Union, List
from collections import defaultdict

from rlkit.core import split_integer
from rlkit.data_management.replay_buffer.env_buffer import EnvReplayBuffer
from rlkit.env_creators.base_env import BaseEnv


class UnifiedReplayBuffer:
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
            self._agent_buffers[a_id] = EnvReplayBuffer(
                max_replay_buffer_size,
                self._observation_space_n[a_id],
                self._action_space_n[a_id],
            )
            self._policy_buffers[p_id][a_id] = self._agent_buffers[a_id]

        self._max_replay_buffer_size = max_replay_buffer_size

    @property
    def agent_buffers(self):
        return self._agent_buffers

    @property
    def policy_buffers(self):
        return self._policy_buffers

    def num_steps_can_sample(
        self, agent_id: str = None, policy_id: str = None, mode="min"
    ) -> int:
        if mode == "min":
            func = min
        elif mode == "max":
            func = max
        if agent_id is None:
            if policy_id is None:
                return func(
                    [
                        buffer.num_steps_can_sample()
                        for buffer in self.agent_buffers.values()
                    ]
                )
            else:
                return func(
                    [
                        buffer.num_steps_can_sample()
                        for buffer in self.policy_buffers[policy_id].values()
                    ]
                )
        else:
            # NOTE(zbzhu): currently one agent can only be controlled by one policy, so we ignore the policy_id here
            return self.agent_buffers[agent_id].num_steps_can_sample()

    def random_batch(
        self,
        batch_size: int,
        agent_id: str = None,
        policy_id: str = None,
        keys: List[str] = None,
        equal_split_agent: bool = False,
    ):
        # OPTIMIZE(zbzhu): make the implementation clear
        if agent_id is None:
            if policy_id is None:
                _agent_ids = self.agent_ids
                _buffers = self.agent_buffers
            else:
                _agent_ids = list(self.policy_buffers[policy_id].keys())
                _buffers = self.policy_buffers[policy_id]

            batch_size_list = split_integer(
                batch_size,
                len(_agent_ids),
                mode="equal" if equal_split_agent else "random",
            )
            return {
                a_id: _buffers[a_id].random_batch(batch_size_list[_idx], keys)
                for _idx, a_id in enumerate(_agent_ids)
            }

        else:
            if policy_id is not None:
                assert agent_id in self.policy_buffers[policy_id].keys()
            return self.agent_buffers[agent_id].random_batch(batch_size, keys)

    def terminate_episode(self):
        for a_id in self.agent_ids:
            self.agent_buffers[a_id].terminate_episode()

    def sample_all_trajs(self, agent_id: str):
        return self.agent_buffers[agent_id].sample_all_trajs()

    def clear(self, agent_id: str):
        self.agent_buffers[agent_id].clear()

    def add_path(self, path_n):
        for a_id in self.agent_ids:
            self.agent_buffers[a_id].add_path(path_n[a_id])

    def add_sample(
        self,
        observation_n,
        action_n,
        reward_n,
        terminal_n,
        next_observation_n,
        **kwargs,
    ):
        for a_id in observation_n.keys():
            self.agent_buffers[a_id].add_sample(
                observation_n[a_id],
                action_n[a_id],
                reward_n[a_id],
                terminal_n[a_id],
                next_observation_n[a_id],
                **{k: v[a_id] if isinstance(v, dict) else v for k, v in kwargs.items()},
            )
