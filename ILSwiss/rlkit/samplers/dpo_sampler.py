import numpy as np

from rlkit.data_management.path_builder import PathBuilder


class DPOPathSampler:
    def __init__(
        self,
        env,
        vec_env,
        policy_n,
        policy_mapping_dict,
        num_steps,
        max_path_length,
        car_num,
        no_terminal=False,
        render=False,
        render_kwargs={},
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.vec_env = vec_env
        self.env_num = vec_env.env_num
        self.wait_num = vec_env.wait_num
        self.car_num = car_num
        self.policy_n = policy_n
        self.policy_mapping_dict = policy_mapping_dict
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
        self.agent_ids = self.env.agent_ids
        self.n_agents = self.env.n_agents

        self.observations_n = self.vec_env.reset()
        self.actions_n = np.array(
            [
                {
                    a_id: self.env.action_space_n[a_id].sample()
                    for a_id in self.agent_ids
                }
                for _ in range(self.env_num)
            ]
        )
        self.pred_observations_n = np.array([None for _ in range(self.env_num)])
        self._ready_env_ids = np.arange(self.env_num)
        self.path_builders = [PathBuilder(self.agent_ids) for _ in range(self.env_num)]

    def obtain_samples(self, num_steps=None):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps

        finished_env_ids = []
        env_finished_car_num = np.zeros(self.env_num)
        while True:
            (
                self.pred_observations_n[self._ready_env_ids],
                self.actions_n[self._ready_env_ids],
            ) = self._get_action_and_info(
                self.observations_n[self._ready_env_ids],
            )

            (
                next_observations_n,
                rewards_n,
                terminals_n,
                env_infos_n,
            ) = self.vec_env.step(
                self.actions_n[self._ready_env_ids].copy(), id=self._ready_env_ids
            )
            self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

            for (
                observation_n,
                action_n,
                reward_n,
                next_observation_n,
                pred_observation_n,
                terminal_n,
                env_info_n,
                env_id,
            ) in zip(
                self.observations_n[self._ready_env_ids],
                self.actions_n[self._ready_env_ids],
                rewards_n,
                next_observations_n,
                self.pred_observations_n[self._ready_env_ids],
                terminals_n,
                env_infos_n,
                self._ready_env_ids,
            ):
                for a_id in observation_n.keys():
                    # some agents may terminate earlier than others
                    if a_id not in next_observation_n.keys():
                        continue
                    self.path_builders[env_id][a_id].add_all(
                        observations=observation_n[a_id],
                        actions=action_n[a_id],
                        rewards=reward_n[a_id],
                        next_observations=next_observation_n[a_id],
                        pred_observations=pred_observation_n[a_id],
                        terminals=terminal_n[a_id],
                        env_infos=env_info_n[a_id],
                    )

            self.observations_n[self._ready_env_ids] = next_observations_n

            terminals_all = [terminal["__all__"] for terminal in terminals_n]
            for env_id, terminal in zip(self._ready_env_ids, terminals_all):
                if terminal or len(self.path_builders[env_id]) >= self.max_path_length:
                    paths.append(self.path_builders[env_id])
                    total_steps += len(self.path_builders[env_id])
                    self.path_builders[env_id] = PathBuilder(self.agent_ids)
                    env_finished_car_num[env_id] += 1
                    if not terminal or not self.vec_env.auto_reset:
                        self.observations_n[env_id] = self.vec_env.reset(id=env_id)[0]
                    if env_finished_car_num[env_id] == self.car_num[env_id]:
                        finished_env_ids.append(env_id)

            self._ready_env_ids = np.array(
                [x for x in self._ready_env_ids if x not in finished_env_ids]
            )

            if len(finished_env_ids) == self.env_num:
                assert len(self._ready_env_ids) == 0
                break

        self._ready_env_ids = np.arange(self.env_num)

        return paths

    def _get_action_and_info(self, observations_n):
        """
        Get an action to take in the environment.
        :param observation_n:
        :return:
        """
        actions_n = [{} for _ in range(len(observations_n))]
        pred_observations_n = [{} for _ in range(len(observations_n))]

        for idx, observation_n in enumerate(observations_n):
            for agent_id in observation_n.keys():
                policy_id = self.policy_mapping_dict[agent_id]
                # OPTIMIZE(zbzhu): can stack all data with same agent_id together and compute once
                (
                    pred_observations_n[idx][agent_id],
                    actions_n[idx][agent_id],
                    _,
                ) = self.policy_n[policy_id].get_action(
                    observation_n[agent_id],
                    return_predicting_obs=True,
                )
        return pred_observations_n, actions_n
