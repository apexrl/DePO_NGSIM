import numpy as np
from rlkit.data_management.replay_buffer.simple_buffer import (
    SimpleReplayBuffer,
    MetaSimpleReplayBuffer,
)
from gym.spaces import Box, Discrete, Tuple, Dict


# XXX(zbzhu): integrate this into unified buffer!!!
class PolicyReplayBuffer:
    def __init__(self, max_replay_buffer_size, env, random_seed=1995):
        self._observation_space_n = env.observation_space_n
        self._action_space_n = env.action_space_n
        self._default_agent_name = env.default_agent_name

        self.policy_buffer = EnvReplayBuffer(
            max_replay_buffer_size,
            self._observation_space_n[self._default_agent_name],
            self._action_space_n[self._default_agent_name],
        )

        self._max_replay_buffer_size = max_replay_buffer_size

    def num_steps_can_sample(self):
        return self.policy_buffer.num_steps_can_sample()

    def random_batch(self, batch_size: int, keys):
        return self.policy_buffer.random_batch(batch_size, keys)

    def terminate_episode(self):
        self.policy_buffer.terminate_episode()

    def sample_all_trajs(self):
        return self.policy_buffer.sample_add_trajs()

    def clear(self):
        self.policy_buffer.clear()

    def add_path(self, path_n, absorbing=False, env=None):
        for a_id in path_n.keys():
            self.policy_buffer.add_path(path_n[a_id], absorbing=absorbing, env=env)

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
            if a_id not in next_observation_n.keys():
                continue
            self.policy_buffer.add_sample(
                observation_n[a_id],
                action_n[a_id],
                reward_n[a_id],
                terminal_n[a_id],
                next_observation_n[a_id],
                **{k: v[a_id] if isinstance(v, dict) else v for k, v in kwargs.items()},
            )


class EnvReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self, max_replay_buffer_size, observation_space, action_space, random_seed=1995
    ):
        """
        :param max_replay_buffer_size:
        :param observation_space:
        :param action_space:
        """
        self._ob_space = observation_space
        self._action_space = action_space
        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            random_seed=random_seed,
        )


class MetaEnvReplayBuffer(MetaSimpleReplayBuffer):
    def __init__(
        self,
        max_rb_size_per_task,
        env,
        extra_obs_dim=0,
        policy_uses_pixels=False,
        policy_uses_task_params=False,
        concat_task_params_to_policy_obs=False,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        assert extra_obs_dim == 0, "I removed the extra_obs_dim thing"
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        super().__init__(
            max_rb_size_per_task=max_rb_size_per_task,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            policy_uses_pixels=policy_uses_pixels,
            policy_uses_task_params=policy_uses_task_params,
            concat_task_params_to_policy_obs=concat_task_params_to_policy_obs,
        )

    def add_path(self, path, task_identifier):
        if isinstance(self._action_space, Discrete):
            action_array = np.eye(self._action_space.n)[path["actions"][:, 0]]
            path["actions"] = action_array
        super(MetaEnvReplayBuffer, self).add_path(path, task_identifier)


def get_dim(space):
    if isinstance(space, Box):
        if len(space.low.shape) > 1:
            return space.low.shape
        return space.low.size
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif isinstance(space, Dict):
        return {k: get_dim(v) for k, v in space.spaces.items()}
    elif hasattr(space, "flat_dim"):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))
