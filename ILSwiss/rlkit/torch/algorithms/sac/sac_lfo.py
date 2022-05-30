from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim

import rlkit.torch.utils.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        inverse_mode="MLE",
        state_predictor_mode="MSE",
        update_both=True,
        train_alpha=False,  # True,
        reward_scale=1.0,
        discount=0.99,
        alpha=1.0,
        policy_lr=1e-3,
        qf_lr=1e-3,
        alpha_lr=3e-4,
        soft_target_tau=1e-2,
        policy_mean_reg_weight=1e-3,
        policy_std_reg_weight=1e-3,
        optimizer_class=optim.Adam,
        beta_1=0.9,
        qscale=False,
        target_entropy=None,
        norm_q=False,
        q_norm_mode="min_max",
        clip_q=False,
        q_clip_eps=0.2,
        ratio_clip_eps=0.2,
        importance_sampling=False,
        agnostic_pg=True,
        ego_obs_idx=None,
        use_ground_truth_inv=False,
        sp_pg_weight=1.0,
        use_target_policy=False,
        target_policy_update_interval=None,
        **kwargs
    ):
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.state_predictor_mode = state_predictor_mode
        self.inverse_mode = inverse_mode

        self.qscale = qscale
        self.norm_q = norm_q
        self.q_norm_mode = q_norm_mode
        self.importance_sampling = importance_sampling

        self.clip_q = clip_q
        self.q_clip_eps = q_clip_eps
        self.ratio_clip_eps = ratio_clip_eps
        self.agnostic_pg = agnostic_pg
        self.sp_pg_weight = sp_pg_weight

        if agnostic_pg:
            print("\nAgnostic PG\n")
        else:
            print("\nNo Agnostic PG\n")

        if qscale:
            print("\nQ SCALE!\n")

        if importance_sampling:
            print("\nIMPORTANCE SAMPLING!\n")

        self.train_alpha = train_alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=train_alpha)
        self.alpha = self.log_alpha.detach().exp()
        assert "env" in kwargs.keys(), "env info should be taken into SAC alpha"

        self.target_entropy = target_entropy
        if target_entropy is None:
            self.target_entropy = -np.prod(kwargs["env"].action_space.shape)

        self.target_qf1 = qf1.copy()
        self.target_qf2 = qf2.copy()

        self.eval_statistics = None

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), lr=policy_lr, betas=(beta_1, 0.999)
        )
        self.update_both = update_both
        if not update_both:
            print("\nSAC UPDATE SP ONLY!\n")
            self.policy_optimizer = optimizer_class(
                self.policy.state_predictor.parameters(),
                lr=policy_lr,
                betas=(beta_1, 0.999),
            )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(), lr=qf_lr, betas=(beta_1, 0.999)
        )
        self.alpha_optimizer = optimizer_class(
            [self.log_alpha], lr=alpha_lr, betas=(beta_1, 0.999)
        )

        self.ego_obs_idx = ego_obs_idx
        self.use_ground_truth_inv = use_ground_truth_inv
        self.use_target_policy = use_target_policy

        if self.use_target_policy:
            self.target_policy = self.policy.copy()
            self.target_policy_update_cnt = 0
            self.target_policy_update_interval = target_policy_update_interval

    def train_step(
        self,
        batch,
        qss=False,
        alpha=None,
        expert_batch=None,
        inv_batch=None,
        state_diff=False,
        forward_model=None,
        cycle_weight=0.0,
        policy_optim_batch_size_from_expert=0,
        update_parameter=True,
    ):
        rewards = self.reward_scale * batch["rewards"]
        terminals = batch["terminals"]
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]

        exp_obs = expert_batch["observations"]
        exp_next_obs = expert_batch["next_observations"]

        if policy_optim_batch_size_from_expert > 0:
            expert_flag = torch.cat(
                [
                    torch.zeros(policy_optim_batch_size_from_expert, 1),
                    torch.ones(policy_optim_batch_size_from_expert, 1),
                ],
                dim=0,
            ).to(ptu.device)
            exp_obs = batch["observations"]
            exp_next_obs = batch["next_observations"]

        input_act = actions
        if qss:
            input_act = next_obs
        """
        QF Loss
        """
        # Only unfreeze parameter of Q
        # for p in itertools.chain(self.qf1.parameters(), self.qf2.parameters()):
        #     p.requires_grad = True
        # for p in self.policy.parameters():
        #     p.requires_grad = False
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(obs, input_act)
        q2_pred = self.qf2(obs, input_act)

        # Make sure policy accounts for squashing functions like tanh correctly!
        if self.use_target_policy:
            next_policy_outputs = self.target_policy(
                next_obs, return_log_prob=True, return_predicting_obs=qss
            )
        else:
            next_policy_outputs = self.policy(
                next_obs, return_log_prob=True, return_predicting_obs=qss
            )

        if qss:
            (
                next_new_next_obs,
                next_new_actions,
                next_policy_mean,
                next_policy_log_std,
                next_log_pi,
            ) = next_policy_outputs[:5]
        else:
            (
                next_new_actions,
                next_policy_mean,
                next_policy_log_std,
                next_log_pi,
            ) = next_policy_outputs[:4]
        next_new_input_act = next_new_actions
        if qss:
            next_new_input_act = next_new_next_obs
        target_qf1_values = self.target_qf1(
            next_obs, next_new_input_act
        )  # do not need grad || it's the shared part of two calculation
        target_qf2_values = self.target_qf2(
            next_obs, next_new_input_act
        )  # do not need grad || it's the shared part of two calculation
        min_target_value = torch.min(target_qf1_values, target_qf2_values)
        q_target = rewards + (1.0 - terminals) * self.discount * (
            min_target_value - self.alpha * next_log_pi
        )  # original implementation has detach
        q_target = q_target.detach()

        qf1_loss = 0.5 * torch.mean((q1_pred - q_target) ** 2)
        qf2_loss = 0.5 * torch.mean((q2_pred - q_target) ** 2)

        qf1_loss.backward()
        qf2_loss.backward()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """
        Policy Loss
        """
        policy_outputs = self.policy(
            obs, return_log_prob=True, return_predicting_obs=qss
        )
        if qss:
            (
                new_pred_obs,
                new_actions,
                policy_mean,
                policy_log_std,
                log_pi,
            ) = policy_outputs[:5]
            new_actions = new_pred_obs
        else:
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        q1_new_acts = self.qf1(obs, new_actions)
        q2_new_acts = self.qf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_acts, q2_new_acts)
        if not self.agnostic_pg:
            q_new_actions = 0

        # self.policy_optimizer.zero_grad()
        # policy_loss = torch.mean(self.alpha * log_pi - q_new_actions)
        if not self.use_ground_truth_inv:
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
        else:
            policy_reg_loss = 0
        # policy_loss = policy_loss + policy_reg_loss

        if update_parameter:
            self.policy_optimizer.zero_grad()

        # !!!!!!!!!!!!!
        actions = self.policy.inverse_dynamic(
            obs[:, self.ego_obs_idx],
            next_obs[:, self.ego_obs_idx],
            True,
        )[0]

        q1_buf_acts = self.qf1(obs, actions)
        q2_buf_acts = self.qf2(obs, actions)

        q_buf_actions = torch.min(q1_buf_acts, q2_buf_acts).detach()
        # q_buf_actions = q_buf_actions - q_buf_actions.min()
        if self.norm_q:
            if self.q_norm_mode == "min_max":
                q_buf_actions = (q_buf_actions - q_buf_actions.min()) / (
                    q_buf_actions.max() - q_buf_actions.min()
                )
            elif self.q_norm_mode == "positive_normal":
                q_buf_actions = (
                    q_buf_actions - q_buf_actions.mean()
                ) / q_buf_actions.std()  # normal
                q_buf_actions = q_buf_actions - q_buf_actions.min()  # enforce positive
            elif self.q_norm_mode == "constant":
                q_buf_actions = 1.0
            else:
                raise NotImplementedError(self.q_norm_mode)
        else:
            q_buf_actions = q_buf_actions - q_buf_actions.min()  # enforce positive

        if self.clip_q:
            q_buf_actions = torch.clamp(
                q_buf_actions,
                min=q_buf_actions.mean() * (1 - self.q_clip_eps),
                max=q_buf_actions.mean() * (1 + self.q_clip_eps),
            )

        # q_buf_actions = (q_buf_actions - q_buf_actions.mean()) / q_buf_actions.std()
        new_label_obs = next_obs[:, self.ego_obs_idx]
        if state_diff:
            new_label_obs = next_obs[:, self.ego_obs_idx] - obs[:, self.ego_obs_idx]
        log_sp = self.policy.state_predictor.get_log_prob(obs, new_label_obs)
        # print("log_sp", log_sp)
        # print("q_buf_actions", q_buf_actions)
        # policy_loss = -1.0 * torch.mean(q_buf_actions * log_sp)

        policy_loss = (
            torch.mean(
                self.alpha * log_pi
                - self.sp_pg_weight * q_buf_actions * log_sp
                - q_new_actions
            )
            + policy_reg_loss
        )

        if alpha is not None:
            if self.state_predictor_mode == "MLE":
                label_obs = exp_next_obs[:, self.ego_obs_idx]
                if state_diff:
                    label_obs = (
                        exp_next_obs[:, self.ego_obs_idx] - exp_obs[:, self.ego_obs_idx]
                    )
                log_prob = self.policy.state_predictor.get_log_prob(exp_obs, label_obs)
                sp_loss = -1.0 * log_prob.mean()
                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Predictor-Log-Likelihood"] = ptu.get_numpy(
                    -1.0 * sp_loss
                )

                # pred_obs = self.policy.state_predictor(exp_obs)
                pred_obs = self.policy.state_predictor(exp_obs, deterministic=True)[0]
                squared_diff = (pred_obs - label_obs) ** 2

                expt_mse = torch.sum(squared_diff, dim=-1).mean()
                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Pred-Expt-MSE"] = ptu.get_numpy(expt_mse)

            if self.state_predictor_mode == "MSE":
                # raise ValueError("not MSE!")
                # pred_obs = self.policy.state_predictor(exp_obs)
                pred_obs = self.policy.state_predictor(exp_obs, deterministic=True)[0]
                label_obs = exp_next_obs[:, self.ego_obs_idx]
                if state_diff:
                    label_obs = (
                        exp_next_obs[:, self.ego_obs_idx] - exp_obs[:, self.ego_obs_idx]
                    )
                squared_diff = (pred_obs - label_obs) ** 2
                sp_loss = torch.sum(squared_diff, dim=-1).mean()

                if self.eval_statistics is None:
                    self.eval_statistics = OrderedDict()
                self.eval_statistics["State-Pred-Expt-MSE"] = ptu.get_numpy(sp_loss)

                agent_label_obs = next_obs[:, self.ego_obs_idx]
                if state_diff:
                    agent_label_obs = (
                        next_obs[:, self.ego_obs_idx] - obs[:, self.ego_obs_idx]
                    )
                agent_pred_obs = self.policy.state_predictor(obs)[0]
                agent_squared_diff = (agent_pred_obs - agent_label_obs) ** 2
                agent_loss = torch.sum(agent_squared_diff, dim=-1).mean()
                self.eval_statistics["State-Pred-Real-MSE"] = ptu.get_numpy(agent_loss)

            if policy_optim_batch_size_from_expert > 0:
                policy_loss = policy_loss + alpha * expert_flag * sp_loss
            else:
                if self.qscale:
                    policy_loss = (
                        policy_loss
                        + alpha * torch.abs(q_buf_actions.max()).detach() * sp_loss
                    )
                else:
                    policy_loss = policy_loss + alpha * sp_loss

        policy_loss.backward()
        if update_parameter:
            self.policy_optimizer.step()

        """
        Update alpha
        """
        if self.train_alpha:
            log_prob = log_pi.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        """
        Update networks
        """
        self._update_target_network()
        if self.use_target_policy:
            self._update_target_policy_network()

        """
        Save some statistics for eval
        """
        if self.eval_statistics is None:
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            self.eval_statistics = OrderedDict()
        self.eval_statistics["Reward Scale"] = self.reward_scale
        self.eval_statistics["QF1 Loss"] = np.mean(ptu.get_numpy(qf1_loss))
        self.eval_statistics["QF2 Loss"] = np.mean(ptu.get_numpy(qf2_loss))
        if self.train_alpha:
            self.eval_statistics["Alpha Loss"] = np.mean(ptu.get_numpy(alpha_loss))
        self.eval_statistics["Policy Loss"] = np.mean(ptu.get_numpy(policy_loss))
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q1 Predictions",
                ptu.get_numpy(q1_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Q2 Predictions",
                ptu.get_numpy(q2_pred),
            )
        )
        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Alpha",
                [ptu.get_numpy(self.alpha)],
            )
        )
        if not self.use_ground_truth_inv:
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Log Pis",
                    ptu.get_numpy(log_pi),
                )
            )
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    "Policy log std",
                    ptu.get_numpy(policy_log_std),
                )
            )

        self.eval_statistics.update(
            create_stats_ordered_dict(
                "Policy mu",
                ptu.get_numpy(policy_mean),
            )
        )

    @property
    def networks(self):
        networks = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        if self.use_target_policy:
            networks.append(self.target_policy)
        return networks

    def _update_target_network(self):
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

    def _update_target_policy_network(self):
        self.target_policy_update_cnt += 1
        if self.target_policy_update_cnt % self.target_policy_update_interval == 0:
            ptu.copy_model_params_from_to(self.policy, self.target_policy)
            self.target_policy_update_cnt = 0

    def get_snapshot(self):
        snapshot = dict(
            log_alpha=self.log_alpha,
            qf1=self.qf1,
            qf2=self.qf2,
            policy=self.policy,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
        if self.use_target_policy:
            snapshot["target_policy"] = self.target_policy
        return snapshot

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        self.eval_statistics = None
