# Copyright (c) 2024, Harsh Gupta
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distillation Agent that inherits from RL-Games A2CAgent.

This module provides a DaGGer-style distillation agent that:
- Inherits all RL-Games infrastructure (multi-GPU, logging, checkpoints)
- Overrides train() to do behavioral cloning instead of PPO
- Loads a teacher model and trains student to mimic it
"""

import os
import time
import torch
import torch.distributed as dist
import numpy as np

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common import common_losses
from rl_games.common.a2c_common import print_statistics, swap_and_flatten01

from typing import Dict


def l2_loss(pred, target):
    """Compute L2 norm loss between predictions and targets."""
    return torch.norm(pred - target, p=2, dim=-1)


def weighted_l2_loss(pred, target, weights):
    """Compute weighted L2 loss."""
    return torch.sum((pred - target) * (weights * (pred - target)), dim=-1) ** 0.5


def adjust_state_dict_keys(checkpoint_state_dict, model_state_dict):
    """Adjust checkpoint keys to match model state dict (handles _orig_mod prefix)."""
    adjusted_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted_state_dict[key] = value
        else:
            # Try with/without _orig_mod prefix
            parts = key.split(".")
            parts.insert(2, "_orig_mod")
            new_key = ".".join(parts)
            
            if new_key in model_state_dict:
                adjusted_state_dict[new_key] = value
            else:
                key_no_orig = key.replace("_orig_mod.", "")
                if key_no_orig in model_state_dict:
                    adjusted_state_dict[key_no_orig] = value
                else:
                    adjusted_state_dict[key] = value
    return adjusted_state_dict


class DistillAgent(A2CAgent):
    """
    Distillation Agent for teacher-student policy transfer.
    
    Inherits from RL-Games A2CAgent to get all infrastructure:
    - Multi-GPU training via DDP
    - TensorBoard logging via self.writer
    - Checkpoint saving/loading
    - game_rewards, game_lengths tracking
    
    Overrides train() to do behavioral cloning:
    - Get teacher actions (no grad)
    - Get student actions (with grad)
    - Compute BC loss (L2 on mus + sigmas)
    - Fixed env assignment: beta fraction use teacher actions
    """
    
    def __init__(self, base_name: str, params: Dict):
        """
        Initialize DistillAgent.
        
        Args:
            base_name: Name for logging/checkpoints
            params: Config dict with:
                - Standard RL-Games params
                - 'distillation': {teacher_cfg, teacher_ckpt, beta, ...}
        """
        # Initialize parent A2CAgent (builds student model, optimizer, etc.)
        super().__init__(base_name, params)
        
        # Get distillation config - THIS IS THE ONLY SOURCE FOR DISTILLATION SETTINGS
        self.distill_config = params["distillation"]

        # Teacher setup
        teacher_cfg = self.distill_config["teacher_cfg"]
        teacher_ckpt_path = self.distill_config["teacher_ckpt"]
        
        # ===== Read ALL settings from distillation: section =====
        self.distill_mode = self.distill_config["mode"]
        if self.distill_mode not in ("dagger", "hybrid"):
            raise ValueError(f"Unsupported distillation.mode='{self.distill_mode}'. Expected 'dagger' or 'hybrid'.")

        # Training hyperparams
        self.distill_lr = self.distill_config["learning_rate"]
        self.distill_grad_norm = self.distill_config["grad_norm"]
        self.max_distill_iters = self.distill_config["max_iterations"]
        # Override save_freq from parent with distillation-specific value
        self.save_freq = self.distill_config["save_frequency"]
        # save_best_after: don't save best until this many epochs (avoid early noise)
        self.save_best_after = self.distill_config["save_best_after"]

        # DaGGer
        self.beta = self.distill_config["beta"]

        # Hybrid (PHP) settings
        self.lambda_d_initial = None
        self.lambda_d_min = None
        self.lambda_d_anneal_epochs = None
        self.hybrid_lr_gate_lambda = None
        self.dagger_loss_coef = float(self.distill_config["dagger_loss_coef"])
        if self.distill_mode == "hybrid":
            self.lambda_d_initial = float(self.distill_config["lambda_d_initial"])
            self.lambda_d_min = float(self.distill_config["lambda_d_min"])
            self.lambda_d_anneal_epochs = int(self.distill_config["lambda_d_anneal_epochs"])
            self.hybrid_lr_gate_lambda = float(self.distill_config["adaptive_lr_gate_lambda_ppo"])
            if self.lambda_d_anneal_epochs <= 0:
                raise ValueError("distillation.lambda_d_anneal_epochs must be > 0 in hybrid mode.")
            if not (0.0 <= self.lambda_d_min <= self.lambda_d_initial <= 1.0):
                raise ValueError(
                    "distillation.lambda_d_min and lambda_d_initial must satisfy "
                    "0.0 <= lambda_d_min <= lambda_d_initial <= 1.0."
                )
            if not (0.0 <= self.hybrid_lr_gate_lambda <= 1.0):
                raise ValueError("distillation.adaptive_lr_gate_lambda_ppo must be in [0, 1].")
            self.lambda_d = self.lambda_d_initial
            self.lambda_ppo = 1.0 - self.lambda_d
        else:
            self.lambda_d = None
            self.lambda_ppo = None

        # Debug depth video recording
        self.debug_depth_video = self.distill_config["debug_depth_video"]
        self.debug_depth_video_fps = self.distill_config["debug_depth_video_fps"]
        self.debug_depth_video_num_frames = self.distill_config["debug_depth_video_num_frames"]
        self.depth_video_frames = []  # Accumulate frames for env 0
        self.depth_video_saved = False  # Only save once
        
        # DAgger uses distillation.* optimizer/scaler settings.
        # Hybrid uses the parent PPO optimizer/scheduler settings from params.config.
        if self.distill_mode == "dagger":
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.distill_lr
            self.last_lr = self.distill_lr
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        
        # Build and load teacher model
        self.teacher_model = self._build_and_load_teacher(teacher_cfg, teacher_ckpt_path)
        self.teacher_model.to(self.ppo_device)
        self.teacher_model.eval()
        
        # Fixed environment assignment for DaGGer.
        # First beta fraction of envs always use teacher, rest always use student.
        num_teacher_envs = int(self.num_actors * self.beta)
        self.teacher_env_mask = torch.zeros(self.num_actors, dtype=torch.bool, device=self.ppo_device)
        if self.distill_mode == "dagger":
            self.teacher_env_mask[:num_teacher_envs] = True
        
        # Check for teacher RNN
        self.is_teacher_rnn = self.teacher_model.is_rnn() if hasattr(self.teacher_model, 'is_rnn') else False
        if self.is_teacher_rnn:
            self.teacher_rnn_states = self.teacher_model.get_default_rnn_state()
            self.teacher_rnn_states = [s.to(self.ppo_device) for s in self.teacher_rnn_states]
        
        # Previous actions for teacher
        self.prev_actions_teacher = torch.zeros(
            (self.num_actors, self.actions_num), 
            dtype=torch.float32, 
            device=self.ppo_device
        )
        
        # Separate tracking for teacher and student envs (split logging, dagger mode).
        self.teacher_game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.teacher_game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.teacher_game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)

        self.student_game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.student_game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.student_game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        
        # Track latest ADR values for console output
        self.last_teacher_adr = None
        self.last_student_adr = None
        
        # Per-reward-term episode sums (we track ourselves since env resets before we can access)
        # Will be initialized in train() after first step when we know the terms
        self.reward_term_names = None
        self.teacher_episode_sums = None  # Dict[str, Tensor] per env
        self.student_episode_sums = None
        self.hybrid_episode_sums = None
        
        # Accumulators for aggregated logging (like IsaacAlgoObserver pattern)
        # These collect values during steps and are flushed to TensorBoard periodically
        self.log_interval = int(self.distill_config.get("log_interval", self.horizon_length))
        if self.log_interval <= 0:
            raise ValueError("distillation.log_interval must be > 0.")
        self.teacher_term_accum = {}  # Dict[term_name, List[float]]
        self.student_term_accum = {}  # Dict[term_name, List[float]]
        self.teacher_adr_accum = []   # List[float]
        self.student_adr_accum = []   # List[float]
        self.hybrid_term_accum = {}   # Dict[term_name, List[float]]
        self.hybrid_adr_accum = []    # List[float]
        self.last_hybrid_adr = None
        
        # RNN training params (inherited from A2CAgent via config)
        # self.seq_length already set by parent from config.get('seq_length', 4)
        # self.zero_rnn_on_done already set by parent from config.get('zero_rnn_on_done', True)
        # For distillation with RNN, we accumulate loss over seq_length steps (truncated BPTT)
        self.accumulated_loss = None  # Will hold accumulated loss for RNN training
        
        print(f"\n{'='*60}")
        print("DistillAgent initialized")
        print(f"  Mode: {self.distill_mode}")
        print(f"  Student model: {self.network}")
        print(f"  Teacher checkpoint: {teacher_ckpt_path}")
        if self.distill_mode == "dagger":
            print(f"  Beta (teacher env ratio): {self.beta}")
            print(f"  Teacher envs: {num_teacher_envs}/{self.num_actors}")
            print(f"  Learning rate: {self.distill_lr}")
            print(f"  Grad norm: {self.distill_grad_norm}")
            print(f"  Max iterations: {self.max_distill_iters}")
        else:
            print(f"  Lambda schedule: initial={self.lambda_d_initial}, min={self.lambda_d_min}, anneal_epochs={self.lambda_d_anneal_epochs}")
            print(f"  Adaptive LR gate: lambda_ppo >= {self.hybrid_lr_gate_lambda}")
            print(f"  PPO learning rate: {self.last_lr}")
            print(f"  Max epochs: {self.max_epochs}")
        print(f"  Save frequency: {self.save_freq}")
        print(f"  Save best after: {self.save_best_after}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Normalize input (RL-Games): {self.normalize_input}")
        print("  Depth augmentation: handled inside student model (depth_resnet_student)")
        if self.debug_depth_video:
            print(f"  Depth video recording: ENABLED (fps={self.debug_depth_video_fps}, num_frames={self.debug_depth_video_num_frames})")
        if self.is_rnn:
            print(f"  RNN enabled:")
            print(f"    seq_length: {self.seq_length}")
        print(f"{'='*60}\n")
    
    def _build_and_load_teacher(self, teacher_cfg, ckpt_path):
        """Build teacher model using teacher config and load weights.
        
        Uses teacher config's obs_groups and queries env for actual dimensions.
        """
        # Teacher normalization behavior should follow the teacher config, not the student config.
        # RL-Games applies normalization internally in the model via `running_mean_std` when enabled.
        teacher_normalize_input = bool(teacher_cfg.get("params", {}).get("config", {}).get("normalize_input", False))
        teacher_normalize_value = bool(teacher_cfg.get("params", {}).get("config", {}).get("normalize_value", False))

        # Get teacher obs_groups from teacher config
        teacher_obs_groups = teacher_cfg['params']['env']['obs_groups']['obs']
        print(f"Teacher obs_groups: {teacher_obs_groups}")
        
        # Query env for dimensions of those groups
        # self.vec_env is RlGamesGpuEnv, .env is RlGamesVecEnvWrapper, .unwrapped is Isaac Lab env
        space = self.vec_env.env.unwrapped.single_observation_space
        teacher_obs_dim = sum(space.get(grp).shape[0] for grp in teacher_obs_groups)
        print(f"Teacher obs_dim: {teacher_obs_dim}")
        
        # Build teacher network using teacher config
        builder = ModelBuilder()
        teacher_network = builder.load(teacher_cfg['params'])
        
        build_config = {
            'actions_num': self.actions_num,
            'input_shape': (teacher_obs_dim,),
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': teacher_normalize_value,
            'normalize_input': teacher_normalize_input,
        }
        
        teacher_model = teacher_network.build(build_config)
        
        # Load weights
        print(f"Loading teacher weights from: {ckpt_path}")
        checkpoint = torch_ext.load_checkpoint(ckpt_path)
        teacher_model.load_state_dict(checkpoint['model'])
        
        # Load RunningMeanStd for teacher if present (DEXTRAH / RL-Games behavior).
        # RL-Games restore semantics:
        #   if normalize_input and 'running_mean_std' in checkpoint: model.running_mean_std.load_state_dict(...)
        # Many valid checkpoints omit running_mean_std; RL-Games simply skips loading in that case.
        if teacher_normalize_input and "running_mean_std" in checkpoint:
            if not hasattr(teacher_model, "running_mean_std"):
                raise AttributeError(
                    "Teacher config has normalize_input=True, but built teacher model has no 'running_mean_std' "
                    "attribute. This likely indicates a mismatch between model type and config."
                )
            teacher_model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            print("[INFO] Loaded teacher running_mean_std from checkpoint.")
        elif teacher_normalize_input and "running_mean_std" not in checkpoint:
            print(
                "[WARNING] Teacher config has normalize_input=True but checkpoint has no 'running_mean_std'. "
                "Proceeding without loading RMS (RL-Games/DEXTRAH default)."
            )
        elif (not teacher_normalize_input) and "running_mean_std" in checkpoint:
            print("[WARNING] Teacher checkpoint contains 'running_mean_std' but teacher normalize_input=False; skipping RMS load.")
        
        print("Teacher model built and loaded successfully!")
        return teacher_model
    
    def train(self):
        """Dispatch to DAgger or Hybrid training loop."""
        if self.distill_mode == "hybrid":
            return self.train_hybrid()
        return self.train_dagger()

    def train_dagger(self):
        """
        Main distillation training loop.
        
        Overrides A2CAgent.train() to do behavioral cloning instead of PPO.
        Keeps the same structure for compatibility with RL-Games infrastructure.
        """
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.perf_counter()
        total_time = 0
        
        # Reset environment
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        
        # Multi-GPU: broadcast student model parameters
        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("Broadcasting student model parameters...")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
        
        # Set models to correct mode
        self.model.train()
        self.teacher_model.eval()
        self._set_apply_depth_aug(True)
        self._set_student_obs_rms_update(True)

        log_counter = 0
        
        print(f"Starting distillation for {self.max_distill_iters} iterations...")
        
        while log_counter < self.max_distill_iters:
            step_start = time.perf_counter()
            
            # Get observations (depth is packed at end of obs, handled by model)
            obs = self._preproc_obs(self.obs['obs'])
            
            # Get teacher observations (may be different from student)
            teacher_obs = self._preproc_obs(self.obs['states'])
            
            # ===== Get teacher actions (no grad) =====
            # Teacher was trained with AMP, so run inside autocast context
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.mixed_precision):
                teacher_batch = {
                    'is_train': False,
                    'obs': teacher_obs,
                    'prev_actions': self.prev_actions_teacher,
                }
                if self.is_teacher_rnn:
                    teacher_batch['rnn_states'] = self.teacher_rnn_states
                    teacher_batch['seq_length'] = 1
                    teacher_batch['rnn_masks'] = None
                
                teacher_res = self.teacher_model(teacher_batch)
                teacher_mus = teacher_res['mus']
                teacher_sigmas = teacher_res['sigmas']
                if self.is_teacher_rnn:
                    self.teacher_rnn_states = teacher_res['rnn_states']
                
                # Match RL-Games training semantics: use the model-sampled actions (no manual clamp).
                # RL-Games controls any clamping/rescaling via config.clip_actions in preprocess_actions().
                teacher_actions = teacher_res.get('actions', None)
                if teacher_actions is None:
                    teacher_distr = torch.distributions.Normal(teacher_mus, teacher_sigmas)
                    teacher_actions = teacher_distr.sample()
            
            # ===== Get student actions (with grad) + Compute loss with AMP =====
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                student_batch = {
                    'is_train': True,
                    'obs': obs,
                    'prev_actions': self.prev_actions if hasattr(self, 'prev_actions') else teacher_actions,
                }
                
                if self.is_rnn:
                    student_batch['rnn_states'] = self.rnn_states if hasattr(self, 'rnn_states') else self.model.get_default_rnn_state()
                    student_batch['seq_length'] = 1
                    student_batch['rnn_masks'] = None
                
                student_res = self.model(student_batch)
                student_mus = student_res['mus']
                student_sigmas = student_res['sigmas']
                if self.is_rnn:
                    self.rnn_states = student_res['rnn_states']
                
                # ===== Compute BC loss =====
                # Weight by inverse sigma^2 for stable training
                weights = 1.0 / (teacher_sigmas.detach() + 1e-6)
                weights = weights ** 2
                
                mu_loss = weighted_l2_loss(student_mus, teacher_mus.detach(), weights).mean()
                sigma_loss = l2_loss(student_sigmas, teacher_sigmas.detach()).mean()
                
                total_loss = mu_loss + sigma_loss
            
            # Sample student actions (outside autocast for stability)
            with torch.no_grad():
                student_distr = torch.distributions.Normal(student_mus.float(), student_sigmas.float())
                student_actions = student_distr.sample()
                # No manual clamp here; let preprocess_actions handle clamping/rescaling if clip_actions=True.
            
            # ===== Backward pass with AMP scaler =====
            # For RNN: accumulate loss over seq_length steps (truncated BPTT)
            # For non-RNN: backward every step
            if self.is_rnn:
                # Accumulate loss
                if self.accumulated_loss is None:
                    self.accumulated_loss = total_loss
                else:
                    self.accumulated_loss = self.accumulated_loss + total_loss
                
                # Only backward every seq_length steps
                if (log_counter + 1) % self.seq_length == 0:
                    self.optimizer.zero_grad()
                    self.scaler.scale(self.accumulated_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.distill_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Detach hidden states to truncate BPTT
                    if self.rnn_states is not None:
                        self.rnn_states = [s.detach() for s in self.rnn_states]
                    
                    # Reset accumulated loss
                    self.accumulated_loss = None
            else:
                # Non-RNN: backward every step
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.distill_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # ===== Choose actions based on fixed env assignment =====
            # teacher_env_mask: True = use teacher, False = use student
            stepping_actions = torch.where(
                self.teacher_env_mask.unsqueeze(-1),
                teacher_actions,
                student_actions
            )
            
            # ===== Step environment =====
            # BEFORE step: grab per-term reward from reward_manager._step_reward (updated in compute())
            # We need to track episode sums ourselves since env resets them before we can access
            isaac_env = self.vec_env.env.unwrapped
            reward_manager = isaac_env.reward_manager
            
            # Initialize per-term tracking on first step
            if self.reward_term_names is None:
                # Use _term_names to ensure order matches _step_reward columns
                self.reward_term_names = reward_manager._term_names
                num_terms = len(self.reward_term_names)
                self.teacher_episode_sums = {name: torch.zeros(self.num_actors, device=self.ppo_device) 
                                            for name in self.reward_term_names}
                self.student_episode_sums = {name: torch.zeros(self.num_actors, device=self.ppo_device) 
                                            for name in self.reward_term_names}
                print(f"[DEBUG] Tracking {num_terms} reward terms: {self.reward_term_names}")
            
            self.obs, rewards, self.dones, infos = self.env_step(stepping_actions)
            
            # Accumulate post-augmentation depth frame for video recording (env 0 only)
            if self.debug_depth_video and not self.depth_video_saved:
                self._accumulate_depth_frame_from_model()
                # Save when we have enough frames
                if len(self.depth_video_frames) >= self.debug_depth_video_num_frames:
                    self._save_depth_video()
            
            # AFTER step: accumulate per-term rewards (step_reward has raw values from this step)
            # step_reward shape: (num_envs, num_terms), _step_reward[:, idx] = value / dt
            # We multiply by dt to get the actual weighted reward contribution
            dt = isaac_env.step_dt
            for idx, term_name in enumerate(self.reward_term_names):
                # _step_reward contains value/dt, multiply back to get actual weighted reward
                step_val = reward_manager._step_reward[:, idx] * dt  # shape: (num_envs,)
                self.teacher_episode_sums[term_name] += step_val * self.teacher_env_mask.float()
                self.student_episode_sums[term_name] += step_val * (~self.teacher_env_mask).float()
            
            # Track rewards and episode lengths (env_step already handles unsqueeze)
            shaped_rewards = self.rewards_shaper(rewards)
            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            
            # Handle done environments (matching parent's logic exactly)
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]  # Handle multi-agent case
            
            # Split done indices by teacher/student mask
            if len(env_done_indices) > 0:
                # Flatten to 1D for consistent indexing
                env_done_flat = env_done_indices.squeeze(-1) if env_done_indices.dim() > 1 else env_done_indices
                teacher_done_mask = self.teacher_env_mask[env_done_flat]
                # Use flattened indices for all downstream indexing
                teacher_done_indices = env_done_flat[teacher_done_mask]
                student_done_indices = env_done_flat[~teacher_done_mask]
                
                # Update teacher game stats
                if len(teacher_done_indices) > 0:
                    self.teacher_game_rewards.update(self.current_rewards[teacher_done_indices])
                    self.teacher_game_shaped_rewards.update(self.current_shaped_rewards[teacher_done_indices])
                    self.teacher_game_lengths.update(self.current_lengths[teacher_done_indices])
                
                # Update student game stats
                if len(student_done_indices) > 0:
                    self.student_game_rewards.update(self.current_rewards[student_done_indices])
                    self.student_game_shaped_rewards.update(self.current_shaped_rewards[student_done_indices])
                    self.student_game_lengths.update(self.current_lengths[student_done_indices])
                
                # Call algo_observer to collect episode info (matching RL-Games pattern)
                # This populates ep_infos which after_print_stats will aggregate and log
                self.algo_observer.process_infos(infos, env_done_indices)
                
                # Accumulate per-term episode rewards (will be logged periodically)
                # Use actual per-env episode lengths from trajectory_manager instead of max
                tm = isaac_env.trajectory_manager
                teacher_ep_lens = tm.t_episode_end[teacher_done_indices] if len(teacher_done_indices) > 0 else None
                student_ep_lens = tm.t_episode_end[student_done_indices] if len(student_done_indices) > 0 else None
                
                for term_name in self.reward_term_names:
                    # Initialize accumulator lists if needed
                    if term_name not in self.teacher_term_accum:
                        self.teacher_term_accum[term_name] = []
                    if term_name not in self.student_term_accum:
                        self.student_term_accum[term_name] = []
                    
                    # Teacher - accumulate instead of immediate log
                    if len(teacher_done_indices) > 0:
                        teacher_sums = self.teacher_episode_sums[term_name][teacher_done_indices]
                        # Normalize each env by its actual episode length, then average
                        val = (teacher_sums / teacher_ep_lens.clamp(min=1e-6)).mean().item()
                        self.teacher_term_accum[term_name].append(val)
                        # Reset for done envs
                        self.teacher_episode_sums[term_name][teacher_done_indices] = 0
                    
                    # Student - accumulate instead of immediate log
                    if len(student_done_indices) > 0:
                        student_sums = self.student_episode_sums[term_name][student_done_indices]
                        # Normalize each env by its actual episode length, then average
                        val = (student_sums / student_ep_lens.clamp(min=1e-6)).mean().item()
                        self.student_term_accum[term_name].append(val)
                        # Reset for done envs
                        self.student_episode_sums[term_name][student_done_indices] = 0
                
                # Accumulate per-env ADR difficulty split by teacher/student
                adr_term = isaac_env.curriculum_manager.cfg.adr.func
                per_env_difficulties = adr_term.current_adr_difficulties  # shape: (num_envs,)
                
                if len(teacher_done_indices) > 0:
                    teacher_adr = per_env_difficulties[teacher_done_indices].mean().item()
                    self.teacher_adr_accum.append(teacher_adr)
                    self.last_teacher_adr = teacher_adr
                
                if len(student_done_indices) > 0:
                    student_adr = per_env_difficulties[student_done_indices].mean().item()
                    self.student_adr_accum.append(student_adr)
                    self.last_student_adr = student_adr
                
                # Reset tracking for done envs (use vectorized multiply like parent)
                not_dones = 1.0 - self.dones.float()
                self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
                self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
                self.current_lengths = self.current_lengths * not_dones
                
                # For RNN: flush accumulated loss before resetting states (like DEXTRAH)
                # This ensures we don't lose gradients from partial sequences
                if self.is_rnn and self.accumulated_loss is not None:
                    self.optimizer.zero_grad()
                    self.scaler.scale(self.accumulated_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.distill_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Detach hidden states
                    if self.rnn_states is not None:
                        self.rnn_states = [s.detach() for s in self.rnn_states]
                    
                    self.accumulated_loss = None
                
                # Reset RNN states for done envs (always reset - episodes are independent)
                if self.is_rnn and self.rnn_states is not None:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = 0.0
                
                if self.is_teacher_rnn and self.teacher_rnn_states is not None:
                    for s in self.teacher_rnn_states:
                        s[:, all_done_indices, :] = 0.0
                
                # Reset prev actions
                self.prev_actions_teacher[env_done_indices] = 0
            
            step_time = time.perf_counter() - step_start
            sum_time = step_time  # For distillation, step_time = sum_time (no separate update phase)
            total_time += sum_time
            
            # Frame calculation (matching RL-Games)
            curr_frames = self.num_actors * self.num_agents
            if self.multi_gpu:
                curr_frames *= self.world_size
            self.frame += curr_frames
            log_counter += 1
            
            should_exit = False
            
            # ===== Logging (matching RL-Games exactly) =====
            if self.global_rank == 0:
                # Diagnostics (matching RL-Games)
                self.diagnostics.epoch(self, current_epoch=log_counter)
                
                # FPS calculations (matching RL-Games exactly)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * step_time  # play_time = step_time for distillation
                
                frame = self.frame // self.num_agents
                
                # Print statistics (matching RL-Games - every 10 iters to reduce spam)
                if log_counter % 10 == 0:
                    print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time,
                                    log_counter, self.max_distill_iters, frame, self.max_frames)
                    
                    # Print distillation stats (teacher/student rewards + loss + ADR)
                    def _get_mean_scalar(meter):
                        m = meter.get_mean()
                        return float(m.item() if hasattr(m, 'item') and m.ndim == 0 else m[0])
                    t_rew = _get_mean_scalar(self.teacher_game_rewards) if self.teacher_game_rewards.current_size > 0 else 0.0
                    s_rew = _get_mean_scalar(self.student_game_rewards) if self.student_game_rewards.current_size > 0 else 0.0
                    # t_adr = f'{self.last_teacher_adr:.2f}' if self.last_teacher_adr is not None else '-'
                    # s_adr = f'{self.last_student_adr:.2f}' if self.last_student_adr is not None else '-'
                    # print(f'  t_rew: {t_rew:.1f} t_adr: {t_adr}  s_rew: {s_rew:.1f} s_adr: {s_adr}  loss: {total_loss.item():.4f}')
                
                # Write stats to TensorBoard (matching RL-Games write_stats)
                self.diagnostics.send_info(self.writer)
                self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
                self.writer.add_scalar('performance/step_inference_time', step_time, frame)
                self.writer.add_scalar('performance/step_time', step_time, frame)
                
                # Distillation-specific losses (similar to RL-Games losses/a_loss, losses/c_loss)
                self.writer.add_scalar('losses/mu_loss', mu_loss.item(), frame)
                self.writer.add_scalar('losses/sigma_loss', sigma_loss.item(), frame)
                self.writer.add_scalar('losses/total_loss', total_loss.item(), frame)
                
                # Info metrics (matching RL-Games)
                self.writer.add_scalar('info/last_lr', self.last_lr, frame)
                self.writer.add_scalar('info/beta', self.beta, frame)
                self.writer.add_scalar('info/epochs', log_counter, frame)
                
                # Call algo_observer (logs Episode/* metrics)
                self.algo_observer.after_print_stats(frame, log_counter, total_time)
                
                # Teacher metrics (separate logging)
                if self.teacher_game_rewards.current_size > 0:
                    teacher_mean_rewards = self.teacher_game_rewards.get_mean()
                    teacher_mean_shaped_rewards = self.teacher_game_shaped_rewards.get_mean()
                    teacher_mean_lengths = self.teacher_game_lengths.get_mean()
                    
                    # Handle scalar vs array (get_mean returns scalar if value_size=1)
                    t_rew = float(teacher_mean_rewards.item() if hasattr(teacher_mean_rewards, 'item') and teacher_mean_rewards.ndim == 0 else teacher_mean_rewards[0])
                    t_shaped = float(teacher_mean_shaped_rewards.item() if hasattr(teacher_mean_shaped_rewards, 'item') and teacher_mean_shaped_rewards.ndim == 0 else teacher_mean_shaped_rewards[0])
                    
                    # Rewards - step, iter, time (matching RL-Games format)
                    self.writer.add_scalar('rewards/teacher_step', t_rew, frame)
                    self.writer.add_scalar('rewards/teacher_iter', t_rew, log_counter)
                    self.writer.add_scalar('rewards/teacher_time', t_rew, total_time)
                    
                    # Shaped rewards
                    self.writer.add_scalar('shaped_rewards/teacher_step', t_shaped, frame)
                    self.writer.add_scalar('shaped_rewards/teacher_iter', t_shaped, log_counter)
                    self.writer.add_scalar('shaped_rewards/teacher_time', t_shaped, total_time)
                    
                    # Episode lengths
                    self.writer.add_scalar('episode_lengths/teacher_step', teacher_mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/teacher_iter', teacher_mean_lengths, log_counter)
                    self.writer.add_scalar('episode_lengths/teacher_time', teacher_mean_lengths, total_time)
                
                # Student metrics (separate logging) - this is the main metric
                s_rew = None  # Define outside for checkpoint use
                if self.student_game_rewards.current_size > 0:
                    mean_rewards = self.student_game_rewards.get_mean()
                    mean_shaped_rewards = self.student_game_shaped_rewards.get_mean()
                    mean_lengths = self.student_game_lengths.get_mean()
                    
                    # Handle scalar vs array (get_mean returns scalar if value_size=1)
                    s_rew = float(mean_rewards.item() if hasattr(mean_rewards, 'item') and mean_rewards.ndim == 0 else mean_rewards[0])
                    s_shaped = float(mean_shaped_rewards.item() if hasattr(mean_shaped_rewards, 'item') and mean_shaped_rewards.ndim == 0 else mean_shaped_rewards[0])
                    self.mean_rewards = s_rew
                    
                    # Rewards - step, iter, time (student metrics)
                    self.writer.add_scalar('rewards/student_step', s_rew, frame)
                    self.writer.add_scalar('rewards/student_iter', s_rew, log_counter)
                    self.writer.add_scalar('rewards/student_time', s_rew, total_time)
                    self.writer.add_scalar('shaped_rewards/student_step', s_shaped, frame)
                    self.writer.add_scalar('shaped_rewards/student_iter', s_shaped, log_counter)
                    self.writer.add_scalar('shaped_rewards/student_time', s_shaped, total_time)
                    
                    self.writer.add_scalar('episode_lengths/student_step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/student_iter', mean_lengths, log_counter)
                    self.writer.add_scalar('episode_lengths/student_time', mean_lengths, total_time)
                    
                    # Save best checkpoint (matching RL-Games exactly)
                    if s_rew > self.last_mean_rewards and log_counter >= self.save_best_after:
                        print('saving next best rewards: ', s_rew)
                        self.last_mean_rewards = s_rew
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        
                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                checkpoint_name = self.config['name'] + '_ep_' + str(log_counter) + '_rew_' + str(s_rew)
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True
                
                # ===== Periodic checkpoint save (every save_freq iterations) =====
                # This is OUTSIDE the student_game_rewards block to ensure it always runs
                if self.save_freq > 0 and log_counter > 0 and log_counter % self.save_freq == 0:
                    # Use mean_rewards if available, else 0
                    rew_str = str(self.mean_rewards) if hasattr(self, 'mean_rewards') and self.mean_rewards > -100000 else '0'
                    checkpoint_name = self.config['name'] + '_ep_' + str(log_counter) + '_rew_' + rew_str
                    print(f'Periodic checkpoint save at iteration {log_counter}')
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))
                
                # ===== Flush accumulated per-term rewards and ADR (every log_interval) =====
                if log_counter % self.log_interval == 0:
                    # Flush per-term episode rewards
                    for term_name in self.teacher_term_accum:
                        if len(self.teacher_term_accum[term_name]) > 0:
                            mean_val = sum(self.teacher_term_accum[term_name]) / len(self.teacher_term_accum[term_name])
                            self.writer.add_scalar(f'Episode/teacher/Episode_Reward/{term_name}', mean_val, frame)
                            self.teacher_term_accum[term_name].clear()
                    
                    for term_name in self.student_term_accum:
                        if len(self.student_term_accum[term_name]) > 0:
                            mean_val = sum(self.student_term_accum[term_name]) / len(self.student_term_accum[term_name])
                            self.writer.add_scalar(f'Episode/student/Episode_Reward/{term_name}', mean_val, frame)
                            self.student_term_accum[term_name].clear()
                    
                    # Flush ADR values
                    if len(self.teacher_adr_accum) > 0:
                        mean_adr = sum(self.teacher_adr_accum) / len(self.teacher_adr_accum)
                        self.writer.add_scalar('Episode/teacher/Curriculum/adr', mean_adr, frame)
                        self.teacher_adr_accum.clear()
                    
                    if len(self.student_adr_accum) > 0:
                        mean_adr = sum(self.student_adr_accum) / len(self.student_adr_accum)
                        self.writer.add_scalar('Episode/student/Curriculum/adr', mean_adr, frame)
                        self.student_adr_accum.clear()
                
                # Check max_epochs exit (matching RL-Games)
                if log_counter >= self.max_distill_iters and self.max_distill_iters != -1:
                    if self.student_game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -float('inf')
                    else:
                        mean_rewards = self.student_game_rewards.get_mean()
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(log_counter) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True
                
                # Check max_frames exit (matching RL-Games)
                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.student_game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -float('inf')
                    else:
                        mean_rewards = self.student_game_rewards.get_mean()
                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True
            
            # Multi-GPU exit sync (matching RL-Games)
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()
            
            if should_exit:
                break
        
        return self.last_mean_rewards, log_counter

    def _set_apply_depth_aug(self, enabled: bool):
        """Toggle explicit depth augmentation in student networks that support it."""
        a2c_network = getattr(self.model, "a2c_network", None)
        if a2c_network is None:
            return
        if hasattr(a2c_network, "apply_depth_aug"):
            a2c_network.apply_depth_aug = bool(enabled)

    def _set_student_obs_rms_update(self, enabled: bool):
        """Toggle student internal obs RMS updates (if present)."""
        a2c_network = getattr(self.model, "a2c_network", None)
        if a2c_network is None:
            return
        if hasattr(a2c_network, "apply_obs_rms_update"):
            a2c_network.apply_obs_rms_update = bool(enabled)
        rms = getattr(a2c_network, "running_mean_std", None)
        if rms is None:
            return
        if enabled:
            rms.train()
        else:
            rms.eval()

    def _replace_depth_with_augmented(self, raw_obs: torch.Tensor) -> torch.Tensor:
        """Replace raw depth tail with post-augmentation depth from the model forward pass."""
        a2c_network = getattr(self.model, "a2c_network", None)
        if a2c_network is None:
            return raw_obs
        if not hasattr(a2c_network, "depth_dim") or not hasattr(a2c_network, "last_depth_img"):
            return raw_obs

        aug_depth = a2c_network.last_depth_img
        if aug_depth is None:
            return raw_obs

        depth_dim = int(a2c_network.depth_dim)
        batch_size = raw_obs.shape[0]
        aug_depth_flat = aug_depth.reshape(batch_size, -1).detach()
        if aug_depth_flat.shape[1] != depth_dim:
            raise RuntimeError(
                f"Augmented depth shape mismatch: expected {depth_dim}, got {aug_depth_flat.shape[1]}."
            )
        return torch.cat([raw_obs[:, :-depth_dim], aug_depth_flat], dim=-1)

    def get_action_values_hybrid(self, obs):
        """Collection-time action/value pass for hybrid mode.

        Runs mostly under eval mode for stable rollout behavior. The only module
        left in train mode is student obs RMS (if present), so normalization
        stats are updated on on-policy rollout data rather than replay data.
        Depth augmentation remains explicitly controlled by apply_depth_aug.
        """
        processed_obs = self._preproc_obs(obs["obs"])
        self.model.eval()
        self._set_apply_depth_aug(True)
        # Update student obs RMS on on-policy rollout data (not replay minibatches).
        self._set_student_obs_rms_update(True)

        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs["states"]
                value_dict = {"is_train": False, "states": states}
                value = self.get_central_value(value_dict)
                res_dict["values"] = value
        return res_dict

    def get_values_hybrid(self, obs):
        """Bootstrap value pass for hybrid mode with rollout-consistent preprocessing."""
        with torch.no_grad():
            if self.has_central_value:
                states = obs["states"]
                self.central_value_net.eval()
                input_dict = {
                    "is_train": False,
                    "states": states,
                    "actions": None,
                    "is_done": self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                self._set_apply_depth_aug(True)
                self._set_student_obs_rms_update(True)
                processed_obs = self._preproc_obs(obs["obs"])
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": processed_obs,
                    "rnn_states": self.rnn_states,
                }
                result = self.model(input_dict)
                value = result["values"]
            return value

    def play_steps_hybrid(self):
        """Collect rollout for hybrid PPO+BC updates."""
        if self.is_rnn:
            raise NotImplementedError("Hybrid mode does not support RNN student models.")

        update_list = self.update_list
        step_time = 0.0
        teacher_mus_steps = []

        self.model.eval()
        self._set_apply_depth_aug(True)
        self.teacher_model.eval()

        for n in range(self.horizon_length):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values_hybrid(self.obs)

            # Store rollout observations with depth replaced by post-augmentation depth.
            obs_with_aug_depth = self._replace_depth_with_augmented(self.obs["obs"])
            self.experience_buffer.update_data("obses", n, obs_with_aug_depth)
            self.experience_buffer.update_data("dones", n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            # Teacher labels for BC term.
            teacher_obs = self._preproc_obs(self.obs["states"])
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.mixed_precision):
                teacher_batch = {
                    "is_train": False,
                    "obs": teacher_obs,
                    "prev_actions": self.prev_actions_teacher,
                }
                if self.is_teacher_rnn:
                    teacher_batch["rnn_states"] = self.teacher_rnn_states
                    teacher_batch["seq_length"] = 1
                    teacher_batch["rnn_masks"] = None

                teacher_res = self.teacher_model(teacher_batch)
                teacher_mus = teacher_res["mus"].detach()
                teacher_mus_steps.append(teacher_mus)

                if self.is_teacher_rnn:
                    self.teacher_rnn_states = teacher_res["rnn_states"]
                if "actions" in teacher_res:
                    self.prev_actions_teacher = teacher_res["actions"].detach()
                else:
                    self.prev_actions_teacher = teacher_mus

            step_time_start = time.perf_counter()
            isaac_env = self.vec_env.env.unwrapped
            reward_manager = isaac_env.reward_manager

            if self.reward_term_names is None or self.hybrid_episode_sums is None:
                self.reward_term_names = reward_manager._term_names
                self.hybrid_episode_sums = {
                    name: torch.zeros(self.num_actors, device=self.ppo_device)
                    for name in self.reward_term_names
                }

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time_end = time.perf_counter()
            step_time += (step_time_end - step_time_start)

            if self.debug_depth_video and not self.depth_video_saved:
                self._accumulate_depth_frame_from_model()
                if len(self.depth_video_frames) >= self.debug_depth_video_num_frames:
                    self._save_depth_video()

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and "time_outs" in infos:
                shaped_rewards += self.gamma * res_dict["values"] * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1

            dt = isaac_env.step_dt
            for idx, term_name in enumerate(self.reward_term_names):
                step_val = reward_manager._step_reward[:, idx] * dt
                self.hybrid_episode_sums[term_name] += step_val

            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            if len(env_done_indices) > 0:
                env_done_flat = env_done_indices.squeeze(-1) if env_done_indices.dim() > 1 else env_done_indices
                tm = isaac_env.trajectory_manager
                ep_lens = tm.t_episode_end[env_done_flat]

                for term_name in self.reward_term_names:
                    if term_name not in self.hybrid_term_accum:
                        self.hybrid_term_accum[term_name] = []
                    ep_sums = self.hybrid_episode_sums[term_name][env_done_flat]
                    val = (ep_sums / ep_lens.clamp(min=1e-6)).mean().item()
                    self.hybrid_term_accum[term_name].append(val)
                    self.hybrid_episode_sums[term_name][env_done_flat] = 0.0

                adr_term = isaac_env.curriculum_manager.cfg.adr.func
                per_env_difficulties = adr_term.current_adr_difficulties
                hybrid_adr = per_env_difficulties[env_done_flat].mean().item()
                self.hybrid_adr_accum.append(hybrid_adr)
                self.last_hybrid_adr = hybrid_adr

            if len(all_done_indices) > 0 and self.is_teacher_rnn and self.teacher_rnn_states is not None:
                for s in self.teacher_rnn_states:
                    s[:, all_done_indices, :] = 0.0
                self.prev_actions_teacher[env_done_indices] = 0.0

            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values_hybrid(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["teacher_mus"] = swap_and_flatten01(torch.stack(teacher_mus_steps, dim=0))
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time
        return batch_dict

    def prepare_dataset(self, batch_dict):
        """Prepare PPO dataset; hybrid mode appends teacher BC targets."""
        if self.distill_mode != "hybrid":
            return super().prepare_dataset(batch_dict)

        obses = batch_dict["obses"]
        returns = batch_dict["returns"]
        dones = batch_dict["dones"]
        values = batch_dict["values"]
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict["rnn_states"] if "rnn_states" in batch_dict else None
        rnn_masks = batch_dict["rnn_masks"] if "rnn_masks" in batch_dict else None

        advantages = returns - values

        if self.normalize_value:
            if self.config["freeze_critic"]:
                self.value_mean_std.eval()
            else:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {
            "old_values": values,
            "old_logp_actions": neglogpacs,
            "advantages": advantages,
            "returns": returns,
            "actions": actions,
            "obs": obses,
            "dones": dones,
            "rnn_states": rnn_states,
            "rnn_masks": rnn_masks,
            "mu": mus,
            "sigma": sigmas,
            "teacher_mus": batch_dict["teacher_mus"],
        }
        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            cv_dict = {
                "old_values": values,
                "advantages": advantages,
                "returns": returns,
                "actions": actions,
                "obs": batch_dict["states"],
                "dones": dones,
                "rnn_masks": rnn_masks,
            }
            self.central_value_net.update_dataset(cv_dict)

    def calc_gradients(self, input_dict):
        """Compute gradients; hybrid mode uses PPO + BC weighted loss."""
        if self.distill_mode != "hybrid":
            return super().calc_gradients(input_dict)

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = self._preproc_obs(input_dict["obs"])
        teacher_mus_batch = input_dict["teacher_mus"]

        lr_mul = 1.0
        curr_e_clip = self.e_clip
        self._set_apply_depth_aug(False)
        self._set_student_obs_rms_update(False)

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, _ = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            ppo_loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            # Paper-faithful hybrid imitation term: plain MSE on expert action means.
            bc_mse_loss = torch.nn.functional.mse_loss(mu, teacher_mus_batch.detach())
            bc_loss = self.dagger_loss_coef * bc_mse_loss

            loss = self.lambda_ppo * ppo_loss + self.lambda_d * bc_loss

            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    if k in self.aux_loss_dict:
                        self.aux_loss_dict[k] = v.detach()
                    else:
                        self.aux_loss_dict[k] = [v.detach()]
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )

        self.train_result = (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.last_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
            ppo_loss.detach(),
            bc_mse_loss.detach(),
        )

    def train_epoch_hybrid(self):
        """One hybrid epoch: rollout collection + PPO/BC minibatch updates."""
        self.vec_env.set_train_info(self.frame, self)
        self.set_train()

        play_time_start = time.perf_counter()
        with torch.no_grad():
            batch_dict = self.play_steps_hybrid()
        play_time_end = time.perf_counter()

        update_time_start = time.perf_counter()
        self.set_train()
        self._set_apply_depth_aug(False)
        # Freeze student obs RMS during replay updates to avoid repeated multi-epoch stat updates.
        self._set_student_obs_rms_update(False)
        rnn_masks = batch_dict["rnn_masks"] if "rnn_masks" in batch_dict else None

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses, c_losses, b_losses, entropies, kls = [], [], [], [], []
        ppo_losses, bc_mse_losses = [], []
        last_lr = self.last_lr
        lr_mul = 1.0

        for mini_ep in range(self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                result = self.train_actor_critic(self.dataset[i])
                (
                    a_loss,
                    c_loss,
                    entropy,
                    kl,
                    last_lr,
                    lr_mul,
                    cmu,
                    csigma,
                    b_loss,
                    ppo_loss,
                    bc_mse_loss,
                ) = result
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                entropies.append(entropy)
                ep_kls.append(kl)
                ppo_losses.append(ppo_loss)
                bc_mse_losses.append(bc_mse_loss)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)

                if self.schedule_type == "legacy":
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    if self.lambda_ppo >= self.hybrid_lr_gate_lambda:
                        self.last_lr, self.entropy_coef = self.scheduler.update(
                            self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                        )
                        self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == "standard" and self.lambda_ppo >= self.hybrid_lr_gate_lambda:
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            self._set_student_obs_rms_update(False)

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            ppo_losses,
            bc_mse_losses,
        )

    def train_hybrid(self):
        """Hybrid PHP-style PPO+BC distillation training loop."""
        if self.is_rnn:
            raise NotImplementedError(
                "Hybrid (PPO+BC) mode does not support RNN student models. Use mode: 'dagger' for RNN."
            )

        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            model_params = [self.model.state_dict()]
            if self.has_central_value:
                model_params.append(self.central_value_net.state_dict())
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
            if self.has_central_value:
                self.central_value_net.load_state_dict(model_params[1])

        while True:
            epoch_num = self.update_epoch()
            anneal_progress = min(1.0, float(epoch_num) / float(self.lambda_d_anneal_epochs))
            lambda_span = self.lambda_d_initial - self.lambda_d_min
            self.lambda_d = self.lambda_d_initial - lambda_span * anneal_progress
            self.lambda_ppo = 1.0 - self.lambda_d

            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
                ppo_losses,
                bc_mse_losses,
            ) = self.train_epoch_hybrid()
            total_time += sum_time
            curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
            self.frame += curr_frames
            frame = self.frame // self.num_agents

            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch=epoch_num)
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                mean_rewards = self.mean_rewards if hasattr(self, "mean_rewards") else -np.inf

                print_statistics(
                    self.print_stats,
                    curr_frames,
                    step_time,
                    scaled_play_time,
                    scaled_time,
                    epoch_num,
                    self.max_epochs,
                    frame,
                    self.max_frames,
                )

                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )

                if len(b_losses) > 0:
                    self.writer.add_scalar("losses/bounds_loss", torch_ext.mean_list(b_losses).item(), frame)
                self.writer.add_scalar("info/lambda_d", self.lambda_d, frame)
                self.writer.add_scalar("info/lambda_ppo", self.lambda_ppo, frame)
                ppo_loss_mean = torch_ext.mean_list(ppo_losses).item()
                bc_mse_loss_mean = torch_ext.mean_list(bc_mse_losses).item()
                bc_loss_pre_coef = bc_mse_loss_mean
                bc_loss_post_coef = self.dagger_loss_coef * bc_loss_pre_coef
                ppo_loss_weighted = self.lambda_ppo * ppo_loss_mean
                bc_loss_weighted = self.lambda_d * bc_loss_post_coef
                hybrid_total_loss = ppo_loss_weighted + bc_loss_weighted

                self.writer.add_scalar("losses/ppo_loss", ppo_loss_mean, frame)
                self.writer.add_scalar("losses/bc_mse_loss", bc_mse_loss_mean, frame)
                self.writer.add_scalar("losses/bc_loss_pre_coef", bc_loss_pre_coef, frame)
                self.writer.add_scalar("losses/bc_loss_post_coef", bc_loss_post_coef, frame)
                self.writer.add_scalar("losses/ppo_loss_weighted", ppo_loss_weighted, frame)
                self.writer.add_scalar("losses/bc_loss_weighted", bc_loss_weighted, frame)
                self.writer.add_scalar("losses/loss_total", hybrid_total_loss, frame)

                if epoch_num % self.log_interval == 0:
                    for term_name, values in self.hybrid_term_accum.items():
                        if len(values) > 0:
                            mean_val = sum(values) / len(values)
                            self.writer.add_scalar(f"Episode/student/Episode_Reward/{term_name}", mean_val, frame)
                            values.clear()

                    if len(self.hybrid_adr_accum) > 0:
                        mean_adr = sum(self.hybrid_adr_accum) / len(self.hybrid_adr_accum)
                        self.writer.add_scalar("Episode/student/Curriculum/adr", mean_adr, frame)
                        self.hybrid_adr_accum.clear()

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()

                    def _metric_item(metric, idx=0):
                        if hasattr(metric, "ndim") and metric.ndim == 0:
                            return float(metric.item())
                        item = metric[idx]
                        return float(item.item() if hasattr(item, "item") else item)

                    mean_reward_0 = _metric_item(mean_rewards, 0)
                    self.mean_rewards = mean_reward_0

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else f"rewards{i}"
                        reward_val = _metric_item(mean_rewards, i)
                        shaped_val = _metric_item(mean_shaped_rewards, i)
                        self.writer.add_scalar(rewards_name + "/step", reward_val, frame)
                        self.writer.add_scalar(rewards_name + "/iter", reward_val, epoch_num)
                        self.writer.add_scalar(rewards_name + "/time", reward_val, total_time)
                        self.writer.add_scalar("shaped_" + rewards_name + "/step", shaped_val, frame)
                        self.writer.add_scalar("shaped_" + rewards_name + "/iter", shaped_val, epoch_num)
                        self.writer.add_scalar("shaped_" + rewards_name + "/time", shaped_val, total_time)

                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar("episode_lengths/iter", mean_lengths, epoch_num)
                    self.writer.add_scalar("episode_lengths/time", mean_lengths, total_time)

                    checkpoint_name = self.config["name"] + "_ep_" + str(epoch_num) + "_rew_" + str(self.mean_rewards)
                    if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, "last_" + checkpoint_name))

                    if self.mean_rewards > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print("saving next best rewards: ", self.mean_rewards)
                        self.last_mean_rewards = self.mean_rewards
                        self.save(os.path.join(self.nn_dir, self.config["name"]))
                        if "score_to_win" in self.config and self.last_mean_rewards > self.config["score_to_win"]:
                            print("Maximum reward achieved. Network won!")
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True

                    mean_rewards = self.mean_rewards

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print("WARNING: Max epochs reached before any env terminated at least once")
                        mean_rewards = -np.inf
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_ep_"
                            + str(epoch_num)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print("WARNING: Max frames reached before any env terminated at least once")
                        mean_rewards = -np.inf
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_frame_"
                            + str(self.frame)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = bool(should_exit_t.item())

            if should_exit:
                return self.last_mean_rewards, epoch_num
    
    def _accumulate_depth_frame_from_model(self):
        """Accumulate POST-AUGMENTATION depth frame for env 0 from student model.
        
        The student model stores `last_depth_img` after applying depth augmentation
        during forward pass. This captures the augmented depth that the model sees.
        """
        # Access post-augmentation depth from student model
        # Model structure: self.model.a2c_network is the Network instance
        try:
            last_depth = self.model.a2c_network.last_depth_img
            if last_depth is None:
                return
        except AttributeError:
            # Model doesn't have last_depth_img (shouldn't happen with updated model)
            return
        
        # last_depth shape: (B, 1, H, W) - get env 0
        depth_img = last_depth[0, 0].cpu().numpy()  # (H, W)
        
        # Convert to RGB using viridis colormap for visualization
        import matplotlib.cm as cm
        depth_colored = cm.viridis(depth_img)[:, :, :3]  # Drop alpha
        depth_rgb = (depth_colored * 255).astype(np.uint8)
        
        self.depth_video_frames.append(depth_rgb)
    
    def _save_depth_video(self):
        """Save accumulated depth frames as video."""
        if len(self.depth_video_frames) == 0:
            return
        
        try:
            import imageio
        except ImportError:
            print("[WARNING] imageio not installed, cannot save depth video. Install with: pip install imageio[ffmpeg]")
            self.depth_video_saved = True
            return
        
        # Create output directory
        video_dir = os.path.join(self.nn_dir, 'depth_videos')
        os.makedirs(video_dir, exist_ok=True)
        
        # Save video
        video_path = os.path.join(video_dir, 'depth_video.mp4')
        
        print(f"\n[DepthVideo] Saving {len(self.depth_video_frames)} frames to {video_path}")
        
        # Use imageio to write video
        writer = imageio.get_writer(video_path, fps=self.debug_depth_video_fps, codec='libx264', quality=8)
        for frame in self.depth_video_frames:
            writer.append_data(frame)
        writer.close()
        
        print(f"[DepthVideo] Saved depth video: {video_path}")
        
        # Mark as saved
        self.depth_video_frames = []
        self.depth_video_saved = True


# Register the agent with RL-Games
def register_distill_agent():
    """Register DistillAgent with RL-Games agent factory."""
    from rl_games.common import object_factory
    from rl_games.torch_runner import Runner
    
    # Get the agent factory
    if hasattr(Runner, 'agent_factory'):
        Runner.agent_factory.register_builder('distill_a2c_continuous', 
                                              lambda **kwargs: DistillAgent(**kwargs))
