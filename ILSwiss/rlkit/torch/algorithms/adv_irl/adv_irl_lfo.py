import numpy as np
from collections import OrderedDict
from typing import Dict, List
import itertools
import random
import gtimer as gt

import torch
import torch.optim as optim
from torch import nn
from torch import autograd
import torch.nn.functional as F

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import dict_list_to_list_dict
from rlkit.torch.algorithms.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder

from tqdm import tqdm


class AdvIRL_LfO(TorchBaseAlgorithm):
    """
    Depending on choice of reward function and size of replay
    buffer this will be:
        - AIRL
        - GAIL (without extra entropy term)
        - FAIRL
        - Discriminator Actor Critic

    I did not implement the reward-wrapping mentioned in
    https://arxiv.org/pdf/1809.02925.pdf though

    Features removed from v1.0:
        - gradient clipping
        - target disc (exponential moving average disc)
        - target policy (exponential moving average policy)
        - disc input noise
    """

    def __init__(
        self,
        mode,  # airl, gail, fairl, or sl
        inverse_mode,  # MLE or MSE
        state_predictor_mode,  # MLE or MSE
        discriminator_n,
        policy_trainer_n,
        expert_replay_buffer,
        state_only=False,
        state_diff=False,
        union=False,
        union_sp=True,
        reward_penelty=False,
        update_weight=False,
        penelty_weight=1.0,
        disc_optim_batch_size=1024,
        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,
        num_update_loops_per_train_call=1,
        num_disc_updates_per_loop_iter=100,
        num_policy_updates_per_loop_iter=100,
        num_train_calls_between_inverse_dynamic_training=1,
        num_state_predictor_updates_per_loop_iter=100,
        max_num_inverse_dynamic_updates_per_loop_iter=None,
        num_inverse_dynamic_updates_per_loop_iter=0,
        num_pretrain_updates=20,
        pretrain_steps_per_epoch=5000,
        disc_lr=1e-3,
        disc_momentum=0.9,
        disc_optimizer_class=optim.Adam,
        state_predictor_lr=1e-3,
        state_predictor_alpha=20,
        state_predictor_momentum=0.0,
        state_predictor_optimizer_class=optim.Adam,
        inverse_dynamic_lr=1e-3,
        inverse_dynamic_beta=0.0,
        inverse_dynamic_momentum=0.0,
        inverse_dynamic_optimizer_class=optim.Adam,
        decay_ratio=1.0,
        use_grad_pen=True,
        use_wgan=False,
        grad_pen_weight=10,
        rew_clip_min=-10,
        rew_clip_max=10,
        valid_ratio=0.2,
        max_valid=5000,
        max_epochs_since_update=5,
        epsilon=0.0,
        min_epsilon=0.0,
        inv_buf_size=1000000,
        epsilon_ratio=1.0,
        rew_shaping=False,
        use_ensemble=False,
        pretrain_inv_num=50,
        share_state_predictor=False,
        use_ground_truth_inv=False,
        share_discriminator=False,
        **kwargs,
    ):
        assert mode in [
            "airl",
            "gail",
            "fairl",
            "gail2",
            "sl",
            "sl-test",
        ], "Invalid adversarial irl algorithm!"
        assert inverse_mode in ["MSE", "MLE", "MAE"], "Invalid bco algorithm!"
        super().__init__(**kwargs)

        self.mode = mode
        self.inverse_mode = inverse_mode
        self.state_predictor_mode = state_predictor_mode
        self.state_only = state_only
        self.state_diff = state_diff
        self.union = union
        self.union_sp = union_sp
        self.reward_penelty = reward_penelty
        self.penelty_weight = penelty_weight
        self.update_weight = update_weight
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.num_inverse_dynamic_updates_per_loop_iter = (
            num_inverse_dynamic_updates_per_loop_iter
        )
        self.epsilon_ratio = epsilon_ratio
        self.rew_shaping = rew_shaping
        self.use_ensemble = use_ensemble
        self.pretrain_inv_num = pretrain_inv_num

        if epsilon > 0:
            print("\nEPSILON GREEDY! {}, RATIO {}\n".format(epsilon, epsilon_ratio))

        print("\nINV BUF SIZE {}!\n".format(inv_buf_size))

        print("\nPRE TRAIN NUM {}!\n".format(pretrain_inv_num))

        # For inv dynamics training's validation
        self.valid_ratio = valid_ratio
        self.max_valid = max_valid
        self._max_epochs_since_update = max_epochs_since_update
        self._epochs_since_update = 0

        self.num_train_calls_between_inverse_dynamic_training = (
            num_train_calls_between_inverse_dynamic_training
        )

        if self.mode in ["sl", "sl-test"]:
            self.union = False
            self.union_sp = False

        self.expert_replay_buffer = expert_replay_buffer

        self.policy_trainer_n = policy_trainer_n
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.discriminator_n = discriminator_n
        self.disc_optimizer_n = {
            policy_id: disc_optimizer_class(
                self.discriminator_n[policy_id].parameters(),
                lr=disc_lr,
                betas=(disc_momentum, 0.999),
            )
            for policy_id in self.policy_ids
        }

        if not use_ground_truth_inv:
            self.inverse_dynamic_optimizer_n = {
                policy_id: inverse_dynamic_optimizer_class(
                    self.exploration_policy_n[policy_id].inverse_dynamic.parameters(),
                    lr=inverse_dynamic_lr,
                    betas=(inverse_dynamic_momentum, 0.999),
                )
                for policy_id in self.policy_ids
            }
        self.state_predictor_alpha = state_predictor_alpha

        self.decay_ratio = decay_ratio

        self.disc_optim_batch_size = disc_optim_batch_size
        self.state_predictor_optim_batch_size = policy_optim_batch_size
        self.inverse_dynamic_optim_batch_size = policy_optim_batch_size

        self.pretrain_steps_per_epoch = pretrain_steps_per_epoch

        print("\nDISC MOMENTUM: %f\n" % disc_momentum)
        print("\nSTATE-PREDICTOR MOMENTUM: %f\n" % state_predictor_momentum)
        print("\nINVERSE-DYNAMIC MOMENTUM: %f\n" % inverse_dynamic_momentum)
        if self.update_weight:
            print("\nUPDATE WEIGHT!\n")
        if self.reward_penelty:
            print("\nREWARD PENELTY!\n")
        if self.rew_shaping:
            print("\nREW SHAPING!\n\n")
        if self.use_ensemble:
            print("\nENSEMBLE INVERSE!\n")

        print(
            f"\nMax num_inverse_dynamic_updates_per_loop_iter: {max_num_inverse_dynamic_updates_per_loop_iter}\n"
        )

        self.bce = nn.BCEWithLogitsLoss()
        self.bce_targets = torch.cat(
            [
                torch.ones(disc_optim_batch_size, 1),
                torch.zeros(disc_optim_batch_size, 1),
            ],
            dim=0,
        )
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)

        self.use_grad_pen = use_grad_pen
        self.use_wgan = use_wgan
        self.grad_pen_weight = grad_pen_weight

        self.num_update_loops_per_train_call = num_update_loops_per_train_call
        self.num_disc_updates_per_loop_iter = num_disc_updates_per_loop_iter
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter
        self.num_state_predictor_updates_per_loop_iter = (
            num_state_predictor_updates_per_loop_iter
        )
        self.max_num_inverse_dynamic_updates_per_loop_iter = (
            max_num_inverse_dynamic_updates_per_loop_iter
        )
        self.num_pretrain_updates = num_pretrain_updates

        self.rew_clip_min = rew_clip_min
        self.rew_clip_max = rew_clip_max
        self.clip_min_rews = rew_clip_min is not None
        self.clip_max_rews = rew_clip_max is not None

        self.disc_eval_statistics = None
        self.policy_eval_statistics = None

        self.ego_obs_idx_n = self.env.ego_obs_idx_n
        self.pred_observations_n = np.array(
            [None for _ in range(self.training_env_num)]
        )

        self.use_ground_truth_inv = use_ground_truth_inv
        self.share_state_predictor = share_state_predictor
        self.share_discriminator = share_discriminator

    def get_batch(
        self,
        batch_size,
        agent_id,
        from_expert,
        keys=None,
    ):
        if from_expert:
            buffer = self.expert_replay_buffer
        else:
            buffer = self.replay_buffer

        batch = buffer.random_batch(batch_size, agent_id, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _end_epoch(self):
        for p_id in self.policy_ids:
            self.policy_trainer_n[p_id].end_epoch()
        self.disc_eval_statistics = None
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_ratio)

        if self.update_weight:
            self.state_predictor_alpha *= self.decay_ratio
        super()._end_epoch()

    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()

        if self.policy_eval_statistics is not None:
            self.eval_statistics.update(self.policy_eval_statistics)
        if "sl" not in self.mode:
            self.eval_statistics.update(self.disc_eval_statistics)
            for p_id in self.policy_ids:
                _statistics = self.policy_trainer_n[p_id].get_eval_statistics()
                if _statistics is not None:
                    for name, data in _statistics.items():
                        self.eval_statistics.update({f"{p_id} {name}": data})

        super().evaluate(epoch, dpo=True, ego_obs_idx_n=self.ego_obs_idx_n)

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        if self.use_ground_truth_inv:
            print("\nUse ground-truth inverse dynamic, skip pretraining ...\n")
            return
        else:
            print("\nPretraining ...\n")
        self._start_new_rollout()

        self._current_path_builder = [
            PathBuilder(self.agent_ids) for _ in range(self.training_env_num)
        ]

        for _ in tqdm(range(self.num_pretrain_updates)):
            # sample data using a random policy
            for steps_this_epoch in range(
                self.pretrain_steps_per_epoch // self.training_env_wait_num
            ):
                (
                    self.pred_observations_n[self._ready_env_ids],
                    self.actions_n[self._ready_env_ids],
                ) = self._get_action_and_info(self.observations_n[self._ready_env_ids])
                if self.render:
                    self.training_env.render()

                (
                    next_obs_n,
                    rewards_n,
                    terminals_n,
                    env_infos_n,
                ) = self.training_env.step(
                    self.actions_n[self._ready_env_ids], self._ready_env_ids
                )
                self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

                if self.no_terminal:
                    terminals_n = [
                        dict(
                            zip(
                                terminal_n.keys(),
                                [False for _ in range(len(terminal_n))],
                            )
                        )
                        for terminal_n in terminals_n
                    ]
                self._n_env_steps_total += self.training_env_wait_num

                self._handle_vec_step(
                    self.observations_n[self._ready_env_ids],
                    self.actions_n[self._ready_env_ids],
                    rewards_n,
                    next_obs_n,
                    self.pred_observations_n[self._ready_env_ids],
                    terminals_n,
                    env_infos_n=env_infos_n,
                    env_ids=self._ready_env_ids,
                )

                terminals_all = [terminal["__all__"] for terminal in terminals_n]

                self.observations_n[self._ready_env_ids] = next_obs_n

                if np.any(terminals_all):
                    end_env_id = self._ready_env_ids[np.where(terminals_all)[0]]
                    self._handle_vec_rollout_ending(end_env_id)
                    if not self.training_env.auto_reset:
                        self.observations_n[end_env_id] = self.training_env.reset(
                            end_env_id
                        )
                elif np.any(
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in range(len(self._ready_env_ids))
                        ]
                    )
                    >= self.max_path_length
                ):
                    env_ind_local = np.where(
                        np.array(
                            [
                                len(self._current_path_builder[i])
                                for i in range(len(self._ready_env_ids))
                            ]
                        )
                        >= self.max_path_length
                    )[0]
                    self._handle_vec_rollout_ending(env_ind_local)
                    self.observations_n[env_ind_local] = self.training_env.reset(
                        env_ind_local
                    )

            for a_id in self.agent_ids:
                self._do_inverse_dynamic_training(
                    -1, a_id, False, valid_ratio=0.0, max_num=self.pretrain_inv_num
                )

    def _get_action_and_info(self, observations_n: List[Dict[str, np.ndarray]]):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        actions_n = [{} for _ in range(len(observations_n))]
        pred_observations_n = [{} for _ in range(len(observations_n))]

        for idx, observation_n in enumerate(observations_n):
            for agent_id in observation_n.keys():
                policy_id = self.policy_mapping_dict[agent_id]
                self.exploration_policy_n[policy_id].set_num_steps_total(
                    self._n_env_steps_total
                )
                if random.random() < self.epsilon:
                    pred_observations_n[idx][agent_id] = None
                    actions_n[idx][agent_id] = self.env.action_space_n[
                        agent_id
                    ].sample()
                else:
                    (
                        pred_observations_n[idx][agent_id],
                        actions_n[idx][agent_id],
                        _,
                    ) = self.exploration_policy_n[policy_id].get_action(
                        observation_n[agent_id],
                        return_predicting_obs=True,
                    )
        return pred_observations_n, actions_n

    def _do_training(self, epoch):
        for a_id in self.agent_ids:
            if (
                not self.use_ground_truth_inv
                and self._n_train_steps_total
                % self.num_train_calls_between_inverse_dynamic_training
                == 0
            ):
                # train inverse dynamics until converged
                self._do_inverse_dynamic_training(epoch, a_id, False)

        for t in range(self.num_update_loops_per_train_call):
            for _ in range(self.num_disc_updates_per_loop_iter):
                if self.share_discriminator:
                    self._do_share_reward_training(epoch)
                else:
                    for a_id in self.agent_ids:
                        self._do_reward_training(epoch, a_id)

            for _ in range(self.num_policy_updates_per_loop_iter):
                if self.share_state_predictor:
                    self._do_share_sp_policy_training(epoch)
                else:
                    for a_id in self.agent_ids:
                        self._do_policy_training(epoch, a_id)

    def _do_inverse_dynamic_training(
        self, epoch, agent_id, use_expert_buffer=False, valid_ratio=None, max_num=None
    ):
        """
        Train the inverse dynamic model
        """
        policy_id = self.policy_mapping_dict[agent_id]

        if valid_ratio is None:
            valid_ratio = self.valid_ratio
        if max_num is None:
            max_num = self.max_num_inverse_dynamic_updates_per_loop_iter

        data_size = self.replay_buffer.num_steps_can_sample(agent_id)  # get all data

        split_idx_sets = range(data_size)

        all_data = self.replay_buffer.get_all(
            agent_id,
            keys=["observations", "actions", "next_observations"],
        )
        all_data = np_to_pytorch_batch(all_data)

        # Split into training and valid sets
        num_valid = min(int(data_size * valid_ratio), self.max_valid)
        num_train = data_size - num_valid
        permutation = np.random.permutation(split_idx_sets)

        train_all_data = {}
        valid_all_data = {}
        for key in all_data:
            # train_all_data[key] = all_data[key][np.concatenate([permutation[num_valid:],unsplit_idx_sets]).astype(np.int32)]
            train_all_data[key] = all_data[key][permutation[num_valid:]]
            valid_all_data[key] = all_data[key][permutation[:num_valid]]

        print("[ Invdynamics ] Training {} | Valid: {}".format(num_train, num_valid))
        idxs = np.arange(num_train)

        if max_num:
            epoch_iter = range(max_num)
        else:
            epoch_iter = itertools.count()

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[idxs]

        batch_size = self.inverse_dynamic_optim_batch_size
        break_train = False
        self.best_valid = 10e7
        self._epochs_since_update = 0

        for inv_train_epoch in epoch_iter:
            idxs = shuffle_rows(idxs)
            if break_train:
                break
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[batch_num * batch_size : (batch_num + 1) * batch_size]
                obs = train_all_data["observations"][batch_idxs][
                    :, self.ego_obs_idx_n[agent_id]
                ]
                acts = train_all_data["actions"][batch_idxs]
                next_obs = train_all_data["next_observations"][batch_idxs][
                    :, self.ego_obs_idx_n[agent_id]
                ]

                if self.inverse_mode == "MLE":
                    log_prob = self.exploration_policy_n[
                        policy_id
                    ].inverse_dynamic.get_log_prob(obs, next_obs, acts)
                    loss = -1.0 * log_prob
                    if self.policy_eval_statistics is None:
                        self.policy_eval_statistics = OrderedDict()
                    self.policy_eval_statistics[
                        f"{policy_id} Inverse-Dynamic-Log-Likelihood"
                    ] = ptu.get_numpy(-1.0 * loss.mean())

                    assert not torch.max(
                        torch.isnan(loss)
                    ), "nan-inverse-dynamic-training, obs: {}, next_obs: {}, acts: {}, log_prob: {}".format(
                        obs,
                        next_obs,
                        acts,
                        log_prob,
                    )

                elif self.inverse_mode == "MSE":
                    pred_acts = self.exploration_policy_n[policy_id].inverse_dynamic(
                        obs, next_obs, deterministic=True
                    )[0]
                    squared_diff = (pred_acts - acts) ** 2
                    loss = torch.sum(squared_diff, dim=-1)
                    if self.policy_eval_statistics is None:
                        self.policy_eval_statistics = OrderedDict()
                    self.policy_eval_statistics[
                        f"{policy_id} Inverse-Dynamic-MSE"
                    ] = ptu.get_numpy(loss.mean())

                loss = torch.mean(loss)
                self.inverse_dynamic_optimizer_n[policy_id].zero_grad()
                loss.backward()
                self.inverse_dynamic_optimizer_n[policy_id].step()

                pred_acts = self.exploration_policy_n[policy_id].inverse_dynamic(
                    obs, next_obs, deterministic=True
                )[0]
                squared_diff = (pred_acts - acts) ** 2
                mse_loss = torch.sum(squared_diff, dim=-1)
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics[
                    f"{policy_id} Inverse-Dynamic-MSE"
                ] = ptu.get_numpy(mse_loss.mean())

            # Do validation
            if num_valid > 0:
                valid_obs = valid_all_data["observations"][
                    :, self.ego_obs_idx_n[agent_id]
                ]
                valid_acts = valid_all_data["actions"]
                valid_next_obs = valid_all_data["next_observations"][
                    :, self.ego_obs_idx_n[agent_id]
                ]
                valid_pred_acts = self.exploration_policy_n[policy_id].inverse_dynamic(
                    valid_obs, valid_next_obs, deterministic=True
                )[0]
                valid_squared_diff = (valid_pred_acts - valid_acts) ** 2
                valid_loss = torch.sum(valid_squared_diff, dim=-1).mean()
                if self.policy_eval_statistics is None:
                    self.policy_eval_statistics = OrderedDict()
                self.policy_eval_statistics[
                    f"{policy_id} Valid-InvDyn-MSE"
                ] = ptu.get_numpy(valid_loss)

                break_train = self.valid_break(
                    inv_train_epoch, ptu.get_numpy(valid_loss)
                )
        print("Final Loss {}".format(loss))

    def valid_break(self, train_epoch, valid_loss):
        updated = False
        current = valid_loss
        best = self.best_valid
        improvement = (best - current) / best
        # print(current, improvement)
        if improvement > 0.01:
            self.best_valid = current
            updated = True
            improvement = (best - current) / best
            # print('epoch {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(train_epoch, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            print(
                "[ Invdynamics ] Breaking at epoch {}: {} epochs since update ({} max)".format(
                    train_epoch,
                    self._epochs_since_update,
                    self._max_epochs_since_update,
                )
            )
            return True
        else:
            return False

    def _do_reward_training(self, epoch, agent_id):
        """
        Train the discriminator
        """
        policy_id = self.policy_mapping_dict[agent_id]

        self.disc_optimizer_n[policy_id].zero_grad()

        keys = ["observations", "next_observations"]

        expert_batch = self.get_batch(
            self.disc_optim_batch_size, agent_id, from_expert=True, keys=keys
        )
        policy_batch = self.get_batch(
            self.disc_optim_batch_size, agent_id, from_expert=False, keys=keys
        )

        expert_obs = expert_batch["observations"]
        policy_obs = policy_batch["observations"]

        expert_next_obs = expert_batch["next_observations"]
        policy_next_obs = policy_batch["next_observations"]

        expert_inputs = [expert_obs, expert_next_obs]
        policy_inputs = [policy_obs, policy_next_obs]

        expert_disc_input = torch.cat(expert_inputs, dim=1)
        policy_disc_input = torch.cat(policy_inputs, dim=1)

        if self.use_wgan:
            expert_logits = self.discriminator_n[policy_id](expert_disc_input)
            policy_logits = self.discriminator_n[policy_id](policy_disc_input)

            disc_ce_loss = -torch.sum(expert_logits) + torch.sum(policy_logits)
        else:
            disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

            disc_logits = self.discriminator_n[policy_id](disc_input)
            disc_preds = (disc_logits > 0).type(disc_logits.data.type())
            disc_ce_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator_n[policy_id](interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        assert not torch.max(
            torch.isnan(disc_total_loss)
        ), "nan-reward-training, disc_ce_loss: {}, disc_grad_pen_loss: {}".format(
            disc_ce_loss, disc_grad_pen_loss
        )
        disc_total_loss.backward()
        self.disc_optimizer_n[policy_id].step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics[f"{policy_id} Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            if not self.use_wgan:
                self.disc_eval_statistics[f"{policy_id} Disc Acc"] = np.mean(
                    ptu.get_numpy(accuracy)
                )
            if self.use_wgan:
                self.disc_eval_statistics[f"{policy_id} Expert D Logits"] = np.mean(
                    ptu.get_numpy(expert_logits)
                )
                self.disc_eval_statistics[f"{policy_id} Policy D Logits"] = np.mean(
                    ptu.get_numpy(policy_logits)
                )
            if self.use_grad_pen:
                self.disc_eval_statistics[f"{policy_id} Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics[f"{policy_id} Grad Pen W"] = np.mean(
                    self.grad_pen_weight
                )

    def _do_share_reward_training(self, epoch):
        """
        Train the discriminator
        """
        policy_id = self.policy_ids[0]

        self.disc_optimizer_n[policy_id].zero_grad()

        keys = ["observations", "next_observations"]

        tot_expert_batch = None
        tot_policy_batch = None

        for agent_id in self.agent_ids:
            expert_batch = self.get_batch(
                self.disc_optim_batch_size, agent_id, from_expert=True, keys=keys
            )
            policy_batch = self.get_batch(
                self.disc_optim_batch_size, agent_id, from_expert=False, keys=keys
            )

            if tot_expert_batch is None:
                tot_expert_batch = expert_batch
            else:
                for key in tot_expert_batch:
                    tot_expert_batch[key] = torch.cat(
                        (tot_expert_batch[key], expert_batch[key]),
                        axis=0,
                    )

            if tot_policy_batch is None:
                tot_policy_batch = expert_batch
            else:
                for key in tot_policy_batch:
                    tot_policy_batch[key] = torch.cat(
                        (tot_policy_batch[key], policy_batch[key]),
                        axis=0,
                    )

        expert_obs = tot_expert_batch["observations"]
        policy_obs = tot_policy_batch["observations"]

        expert_next_obs = tot_expert_batch["next_observations"]
        policy_next_obs = tot_policy_batch["next_observations"]

        expert_inputs = [expert_obs, expert_next_obs]
        policy_inputs = [policy_obs, policy_next_obs]

        expert_disc_input = torch.cat(expert_inputs, dim=1)
        policy_disc_input = torch.cat(policy_inputs, dim=1)

        if self.use_wgan:
            expert_logits = self.discriminator_n[policy_id](expert_disc_input)
            policy_logits = self.discriminator_n[policy_id](policy_disc_input)

            disc_ce_loss = -torch.sum(expert_logits) + torch.sum(policy_logits)
        else:
            disc_input = torch.cat([expert_disc_input, policy_disc_input], dim=0)

            disc_logits = self.discriminator_n[policy_id](disc_input)
            disc_preds = (disc_logits > 0).type(disc_logits.data.type())
            disc_ce_loss = self.bce(disc_logits, self.bce_targets)
            accuracy = (disc_preds == self.bce_targets).type(torch.FloatTensor).mean()

        if self.use_grad_pen:
            eps = ptu.rand(expert_obs.size(0), 1)
            eps.to(ptu.device)

            interp_obs = eps * expert_disc_input + (1 - eps) * policy_disc_input
            interp_obs = interp_obs.detach()
            interp_obs.requires_grad_(True)

            gradients = autograd.grad(
                outputs=self.discriminator_n[policy_id](interp_obs).sum(),
                inputs=[interp_obs],
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )
            total_grad = gradients[0]

            # GP from Gulrajani et al.
            gradient_penalty = ((total_grad.norm(2, dim=1) - 1) ** 2).mean()
            disc_grad_pen_loss = gradient_penalty * self.grad_pen_weight

            # # GP from Mescheder et al.
            # gradient_penalty = (total_grad.norm(2, dim=1) ** 2).mean()
            # disc_grad_pen_loss = gradient_penalty * 0.5 * self.grad_pen_weight
        else:
            disc_grad_pen_loss = 0.0

        disc_total_loss = disc_ce_loss + disc_grad_pen_loss
        assert not torch.max(
            torch.isnan(disc_total_loss)
        ), "nan-reward-training, disc_ce_loss: {}, disc_grad_pen_loss: {}".format(
            disc_ce_loss, disc_grad_pen_loss
        )
        disc_total_loss.backward()
        self.disc_optimizer_n[policy_id].step()

        """
        Save some statistics for eval
        """
        if self.disc_eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.disc_eval_statistics = OrderedDict()

            self.disc_eval_statistics["Shared Disc CE Loss"] = np.mean(
                ptu.get_numpy(disc_ce_loss)
            )
            if not self.use_wgan:
                self.disc_eval_statistics["Shared Disc Acc"] = np.mean(
                    ptu.get_numpy(accuracy)
                )
            if self.use_wgan:
                self.disc_eval_statistics["Shared Expert D Logits"] = np.mean(
                    ptu.get_numpy(expert_logits)
                )
                self.disc_eval_statistics["Shared Policy D Logits"] = np.mean(
                    ptu.get_numpy(policy_logits)
                )
            if self.use_grad_pen:
                self.disc_eval_statistics["Shared Grad Pen"] = np.mean(
                    ptu.get_numpy(gradient_penalty)
                )
                self.disc_eval_statistics["Shared Grad Pen W"] = np.mean(
                    self.grad_pen_weight
                )

    def _do_share_sp_policy_training(self, epoch):
        self.policy_trainer_n[self.policy_ids[0]].policy_optimizer.zero_grad()

        for agent_id in self.agent_ids:
            policy_id = self.policy_mapping_dict[agent_id]
            if self.policy_optim_batch_size_from_expert > 0:
                policy_batch_from_policy_buffer = self.get_batch(
                    self.policy_optim_batch_size
                    - self.policy_optim_batch_size_from_expert,
                    agent_id=agent_id,
                    from_expert=False,
                )
                policy_batch_from_expert_buffer = self.get_batch(
                    self.policy_optim_batch_size_from_expert,
                    agent_id=agent_id,
                    from_expert=True,
                )
                policy_batch = {}
                for k in policy_batch_from_policy_buffer:
                    policy_batch[k] = torch.cat(
                        [
                            policy_batch_from_policy_buffer[k],
                            policy_batch_from_expert_buffer[k],
                        ],
                        dim=0,
                    )
            else:
                policy_batch = self.get_batch(
                    self.policy_optim_batch_size, agent_id=agent_id, from_expert=False
                )

            obs = policy_batch["observations"]
            next_obs = policy_batch["next_observations"]

            policy_inputs = [obs, next_obs]

            self.discriminator_n[policy_id].eval()
            disc_input = torch.cat(policy_inputs, dim=1)
            disc_logits = self.discriminator_n[policy_id](disc_input).detach()
            self.discriminator_n[policy_id].train()

            # compute the reward using the algorithm
            if self.mode == "airl":
                # If you compute log(D) - log(1-D) then you just get the logits
                policy_batch["rewards"] = disc_logits
            elif self.mode == "gail":
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=1
                )  # F.softplus(disc_logits, beta=-1)
            elif self.mode == "gail2":
                policy_batch["rewards"] = F.softplus(
                    disc_logits, beta=-1
                )  # F.softplus(disc_logits, beta=-1)
            else:  # fairl
                policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

            if self.reward_penelty:
                agent_pred_obs = self.exploration_policy_n[policy_id].state_predictor(
                    obs
                )
                pred_mse = (agent_pred_obs - next_obs) ** 2
                pred_mse = torch.sum(pred_mse, axis=-1, keepdim=True)
                reward_penelty = self.penelty_weight * pred_mse
                policy_batch["rewards"] -= reward_penelty

                self.disc_eval_statistics[f"{agent_id} Penelty Rew Mean"] = np.mean(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics[f"{agent_id} Penelty Rew Std"] = np.std(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics[f"{agent_id} Penelty Rew Max"] = np.max(
                    ptu.get_numpy(reward_penelty)
                )
                self.disc_eval_statistics[f"{agent_id} Penelty Rew Min"] = np.min(
                    ptu.get_numpy(reward_penelty)
                )

            if self.clip_max_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], max=self.rew_clip_max
                )
            if self.clip_min_rews:
                policy_batch["rewards"] = torch.clamp(
                    policy_batch["rewards"], min=self.rew_clip_min
                )
                if self.rew_shaping:
                    policy_batch["rewards"] -= self.rew_clip_min

            # policy optimization step
            exp_keys = ["observations", "next_observations"]

            expert_batch = self.get_batch(
                self.state_predictor_optim_batch_size,
                keys=exp_keys,
                agent_id=agent_id,
                from_expert=True,
            )

            self.policy_trainer_n[policy_id].train_step(
                policy_batch,
                alpha=self.state_predictor_alpha,
                expert_batch=expert_batch,
                policy_optim_batch_size_from_expert=self.policy_optim_batch_size_from_expert,
                state_diff=self.state_diff,
                update_parameter=False,
            )

            self.disc_eval_statistics[f"{agent_id} Disc Rew Mean"] = np.mean(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics[f"{agent_id} Disc Rew Std"] = np.std(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics[f"{agent_id} Disc Rew Max"] = np.max(
                ptu.get_numpy(policy_batch["rewards"])
            )
            self.disc_eval_statistics[f"{agent_id} Disc Rew Min"] = np.min(
                ptu.get_numpy(policy_batch["rewards"])
            )

        self.policy_trainer_n[self.policy_ids[0]].policy_optimizer.step()

    def _do_policy_training(self, epoch, agent_id):

        policy_id = self.policy_mapping_dict[agent_id]

        if self.policy_optim_batch_size_from_expert > 0:
            policy_batch_from_policy_buffer = self.get_batch(
                self.policy_optim_batch_size - self.policy_optim_batch_size_from_expert,
                agent_id=agent_id,
                from_expert=False,
            )
            policy_batch_from_expert_buffer = self.get_batch(
                self.policy_optim_batch_size_from_expert,
                agent_id=agent_id,
                from_expert=True,
            )
            policy_batch = {}
            for k in policy_batch_from_policy_buffer:
                policy_batch[k] = torch.cat(
                    [
                        policy_batch_from_policy_buffer[k],
                        policy_batch_from_expert_buffer[k],
                    ],
                    dim=0,
                )
        else:
            policy_batch = self.get_batch(
                self.policy_optim_batch_size, agent_id=agent_id, from_expert=False
            )

        obs = policy_batch["observations"]
        next_obs = policy_batch["next_observations"]

        policy_inputs = [obs, next_obs]

        self.discriminator_n[policy_id].eval()
        disc_input = torch.cat(policy_inputs, dim=1)
        disc_logits = self.discriminator_n[policy_id](disc_input).detach()
        self.discriminator_n[policy_id].train()

        # compute the reward using the algorithm
        if self.mode == "airl":
            # If you compute log(D) - log(1-D) then you just get the logits
            policy_batch["rewards"] = disc_logits
        elif self.mode == "gail":
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=1
            )  # F.softplus(disc_logits, beta=-1)
        elif self.mode == "gail2":
            policy_batch["rewards"] = F.softplus(
                disc_logits, beta=-1
            )  # F.softplus(disc_logits, beta=-1)
        else:  # fairl
            policy_batch["rewards"] = torch.exp(disc_logits) * (-1.0 * disc_logits)

        if self.reward_penelty:
            agent_pred_obs = self.exploration_policy_n[policy_id].state_predictor(obs)
            pred_mse = (agent_pred_obs - next_obs) ** 2
            pred_mse = torch.sum(pred_mse, axis=-1, keepdim=True)
            reward_penelty = self.penelty_weight * pred_mse
            policy_batch["rewards"] -= reward_penelty

            self.disc_eval_statistics[f"{agent_id} Penelty Rew Mean"] = np.mean(
                ptu.get_numpy(reward_penelty)
            )
            self.disc_eval_statistics[f"{agent_id} Penelty Rew Std"] = np.std(
                ptu.get_numpy(reward_penelty)
            )
            self.disc_eval_statistics[f"{agent_id} Penelty Rew Max"] = np.max(
                ptu.get_numpy(reward_penelty)
            )
            self.disc_eval_statistics[f"{agent_id} Penelty Rew Min"] = np.min(
                ptu.get_numpy(reward_penelty)
            )

        if self.clip_max_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], max=self.rew_clip_max
            )
        if self.clip_min_rews:
            policy_batch["rewards"] = torch.clamp(
                policy_batch["rewards"], min=self.rew_clip_min
            )
            if self.rew_shaping:
                policy_batch["rewards"] -= self.rew_clip_min

        # policy optimization step
        exp_keys = ["observations", "next_observations"]

        expert_batch = self.get_batch(
            self.state_predictor_optim_batch_size,
            keys=exp_keys,
            agent_id=agent_id,
            from_expert=True,
        )
        self.policy_trainer_n[policy_id].train_step(
            policy_batch,
            alpha=self.state_predictor_alpha,
            expert_batch=expert_batch,
            policy_optim_batch_size_from_expert=self.policy_optim_batch_size_from_expert,
            state_diff=self.state_diff,
        )

        self.disc_eval_statistics[f"{agent_id} Disc Rew Mean"] = np.mean(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Std"] = np.std(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Max"] = np.max(
            ptu.get_numpy(policy_batch["rewards"])
        )
        self.disc_eval_statistics[f"{agent_id} Disc Rew Min"] = np.min(
            ptu.get_numpy(policy_batch["rewards"])
        )

    def _handle_step(
        self,
        observation_n,
        action_n,
        reward_n,
        next_observation_n,
        pred_observation_n,
        terminal_n,
        env_info_n,
        env_id=None,
        add_buf=True,
        path_builder=True,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        if path_builder:
            assert env_id is not None
            for a_id in observation_n.keys():
                # some agents may terminate earlier than others
                if a_id not in next_observation_n.keys():
                    continue
                self._current_path_builder[env_id][a_id].add_all(
                    observations=observation_n[a_id],
                    actions=action_n[a_id],
                    rewards=reward_n[a_id],
                    next_observations=next_observation_n[a_id],
                    pred_observations=pred_observation_n[a_id],
                    terminals=terminal_n[a_id],
                    env_infos=env_info_n[a_id],
                )

        if add_buf:
            self.replay_buffer.add_sample(
                observation_n=observation_n,
                action_n=action_n,
                reward_n=reward_n,
                terminal_n=terminal_n,
                next_observation_n=next_observation_n,
                pred_observation_n=pred_observation_n,
            )

    def _handle_path(self, path, env_id=None):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            pred_ob_n,
            terminal_n,
            env_info_n,
        ) in zip(
            *map(
                dict_list_to_list_dict,
                [
                    path.get_all_agent_dict("observations"),
                    path.get_all_agent_dict("actions"),
                    path.get_all_agent_dict("rewards"),
                    path.get_all_agent_dict("next_observations"),
                    path.get_all_agent_dict("pred_observations"),
                    path.get_all_agent_dict("terminals"),
                    path.get_all_agent_dict("env_infos"),
                ],
            )
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                pred_ob_n,
                terminal_n,
                env_info_n=env_info_n,
                path_builder=False,
                env_id=env_id,
            )

    def _handle_vec_step(
        self,
        observations_n: List,
        actions_n: List,
        rewards_n: List,
        next_observations_n: List,
        pred_observations_n: List,
        terminals_n: List,
        env_infos_n: List,
        env_ids: List,
    ):
        """
        Implement anything that needs to happen after every step under vec envs
        :return:
        """
        for (
            ob_n,
            action_n,
            reward_n,
            next_ob_n,
            pred_ob_n,
            terminal_n,
            env_info_n,
            env_id,
        ) in zip(
            observations_n,
            actions_n,
            rewards_n,
            next_observations_n,
            pred_observations_n,
            terminals_n,
            env_infos_n,
            env_ids,
        ):
            self._handle_step(
                ob_n,
                action_n,
                reward_n,
                next_ob_n,
                pred_ob_n,
                terminal_n,
                env_info_n=env_info_n,
                env_id=env_id,
                add_buf=False,
            )

    @property
    def networks_n(self):
        return {
            p_id: [self.discriminator_n[p_id]] + self.policy_trainer_n[p_id].networks
            for p_id in self.policy_ids
        }

    def get_epoch_snapshot(self, epoch):
        snapshot = dict(epoch=epoch)
        for p_id in self.policy_ids:
            snapshot[p_id] = self.policy_trainer_n[p_id].get_snapshot()
            snapshot[p_id].update(disc=self.discriminator_n[p_id])
        return snapshot

    def to(self, device):
        self.bce.to(ptu.device)
        self.bce_targets = self.bce_targets.to(ptu.device)
        super().to(device)

    def start_training(self, start_epoch=0):
        self._start_new_rollout()

        self._current_path_builder = [
            PathBuilder(self.agent_ids) for _ in range(self.training_env_num)
        ]

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self._start_epoch(epoch)
            for steps_this_epoch in tqdm(
                range(self.num_env_steps_per_epoch // self.training_env_wait_num),
                unit_scale=self.training_env_wait_num,
            ):
                (
                    self.pred_observations_n[self._ready_env_ids],
                    self.actions_n[self._ready_env_ids],
                ) = self._get_action_and_info(self.observations_n[self._ready_env_ids])

                for action_n in self.actions_n[self._ready_env_ids]:
                    for a_id, action in action_n.items():
                        if type(action) is tuple:
                            action_n[a_id] = action_n[a_id][0]

                if self.render:
                    self.training_env.render()

                (
                    next_obs_n,
                    rewards_n,
                    terminals_n,
                    env_infos_n,
                ) = self.training_env.step(
                    self.actions_n[self._ready_env_ids], self._ready_env_ids
                )
                self._ready_env_ids = np.array([i["env_id"] for i in env_infos_n])

                if self.no_terminal:
                    terminals_n = [
                        dict(
                            zip(
                                terminal_n.keys(),
                                [False for _ in range(len(terminal_n))],
                            )
                        )
                        for terminal_n in terminals_n
                    ]
                self._n_env_steps_total += self.training_env_wait_num

                self._handle_vec_step(
                    self.observations_n[self._ready_env_ids],
                    self.actions_n[self._ready_env_ids],
                    rewards_n,
                    next_obs_n,
                    self.pred_observations_n[self._ready_env_ids],
                    terminals_n,
                    env_ids=self._ready_env_ids,
                    env_infos_n=env_infos_n,
                )

                terminals_all = [terminal["__all__"] for terminal in terminals_n]

                self.observations_n[self._ready_env_ids] = next_obs_n

                if np.any(terminals_all):
                    end_env_id = self._ready_env_ids[np.where(terminals_all)[0]]
                    self._handle_vec_rollout_ending(end_env_id)
                    if not self.training_env.auto_reset:
                        self.observations_n[end_env_id] = self.training_env.reset(
                            end_env_id
                        )
                elif np.any(
                    np.array(
                        [
                            len(self._current_path_builder[i])
                            for i in range(len(self._ready_env_ids))
                        ]
                    )
                    >= self.max_path_length
                ):
                    env_ind_local = np.where(
                        np.array(
                            [
                                len(self._current_path_builder[i])
                                for i in range(len(self._ready_env_ids))
                            ]
                        )
                        >= self.max_path_length
                    )[0]
                    self._handle_vec_rollout_ending(env_ind_local)
                    self.observations_n[env_ind_local] = self.training_env.reset(
                        env_ind_local
                    )

                if (
                    self._n_env_steps_total - self._n_prev_train_env_steps
                ) >= self.num_steps_between_train_calls:
                    gt.stamp("sample")
                    self._try_to_train(epoch)
                    gt.stamp("train")

            gt.stamp("sample")
            self._try_to_eval(epoch)
            gt.stamp("eval")
            self._end_epoch()
