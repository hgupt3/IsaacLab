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
from rl_games.common.a2c_common import print_statistics

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
        self.distill_config = params.get('distillation', {})
        
        # Teacher setup
        teacher_cfg = self.distill_config.get('teacher_cfg')
        teacher_ckpt_path = self.distill_config.get('teacher_ckpt')
        
        if teacher_cfg is None or teacher_ckpt_path is None:
            raise ValueError("distillation.teacher_cfg and distillation.teacher_ckpt must be provided")
        
        # ===== Read ALL settings from distillation: section =====
        # Training hyperparams
        self.distill_lr = self.distill_config.get('learning_rate', 1e-4)
        self.distill_grad_norm = self.distill_config.get('grad_norm', 1.0)
        self.max_distill_iters = self.distill_config.get('max_iterations', 1_000_000)
        # Override save_freq from parent with distillation-specific value
        self.save_freq = self.distill_config.get('save_frequency', 5000)
        # save_best_after: don't save best until this many epochs (avoid early noise)
        self.save_best_after = self.distill_config.get('save_best_after', 0)
        
        # DaGGer
        self.beta = self.distill_config.get('beta', 0.5)
        
        # Value distillation
        self.use_value_distillation = self.distill_config.get('value_distillation', True)
        
        # Infrastructure
        self.distill_mixed_precision = self.distill_config.get('mixed_precision', True)
        self.distill_normalize_input = self.distill_config.get('normalize_input', True)
        # multi_gpu is handled by script via --distributed flag
        
        # Debug depth video recording
        self.debug_depth_video = self.distill_config.get('debug_depth_video', False)
        self.debug_depth_video_fps = self.distill_config.get('debug_depth_video_fps', 30)
        self.debug_depth_video_num_frames = self.distill_config.get('debug_depth_video_num_frames', 1000)
        self.depth_video_frames = []  # Accumulate frames for env 0
        self.depth_video_saved = False  # Only save once
        
        # Set learning rate in optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.distill_lr
        self.last_lr = self.distill_lr
        
        # Setup AMP scaler for mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.distill_mixed_precision)
        
        # Build and load teacher model
        self.teacher_model = self._build_and_load_teacher(teacher_cfg, teacher_ckpt_path)
        self.teacher_model.to(self.ppo_device)
        self.teacher_model.eval()
        
        # Fixed environment assignment for DaGGer
        # First beta fraction of envs always use teacher, rest always use student
        num_teacher_envs = int(self.num_actors * self.beta)
        self.teacher_env_mask = torch.zeros(self.num_actors, dtype=torch.bool, device=self.ppo_device)
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
        
        # Separate tracking for teacher and student envs (split logging)
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
        
        # Accumulators for aggregated logging (like IsaacAlgoObserver pattern)
        # These collect values during steps and are flushed to TensorBoard periodically
        self.log_interval = 36  # Log every N iterations (matches horizon_length)
        self.teacher_term_accum = {}  # Dict[term_name, List[float]]
        self.student_term_accum = {}  # Dict[term_name, List[float]]
        self.teacher_adr_accum = []   # List[float]
        self.student_adr_accum = []   # List[float]
        
        # RNN training params (inherited from A2CAgent via config)
        # self.seq_length already set by parent from config.get('seq_length', 4)
        # self.zero_rnn_on_done already set by parent from config.get('zero_rnn_on_done', True)
        # For distillation with RNN, we accumulate loss over seq_length steps (truncated BPTT)
        self.accumulated_loss = None  # Will hold accumulated loss for RNN training
        
        print(f"\n{'='*60}")
        print("DistillAgent initialized")
        print(f"  Student model: {self.network}")
        print(f"  Teacher checkpoint: {teacher_ckpt_path}")
        print(f"  Beta (teacher env ratio): {self.beta}")
        print(f"  Teacher envs: {num_teacher_envs}/{self.num_actors}")
        print(f"  Learning rate: {self.distill_lr}")
        print(f"  Grad norm: {self.distill_grad_norm}")
        print(f"  Max iterations: {self.max_distill_iters}")
        print(f"  Save frequency: {self.save_freq}")
        print(f"  Save best after: {self.save_best_after}")
        print(f"  Mixed precision: {self.distill_mixed_precision}")
        print(f"  Normalize input: {self.distill_normalize_input}")
        print(f"  Value distillation: {self.use_value_distillation}")
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
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.distill_mixed_precision):
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
                teacher_values = teacher_res.get('values', None)  # For value distillation
                
                if self.is_teacher_rnn:
                    self.teacher_rnn_states = teacher_res['rnn_states']
                
                # Match RL-Games training semantics: use the model-sampled actions (no manual clamp).
                # RL-Games controls any clamping/rescaling via config.clip_actions in preprocess_actions().
                teacher_actions = teacher_res.get('actions', None)
                if teacher_actions is None:
                    teacher_distr = torch.distributions.Normal(teacher_mus, teacher_sigmas)
                    teacher_actions = teacher_distr.sample()
            
            # ===== Get student actions (with grad) + Compute loss with AMP =====
            with torch.amp.autocast('cuda', enabled=self.distill_mixed_precision):
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
                student_values = student_res.get('values', None)  # For value distillation
                
                if self.is_rnn:
                    self.rnn_states = student_res['rnn_states']
                
                # ===== Compute BC loss =====
                # Weight by inverse sigma^2 for stable training
                weights = 1.0 / (teacher_sigmas.detach() + 1e-6)
                weights = weights ** 2
                
                mu_loss = weighted_l2_loss(student_mus, teacher_mus.detach(), weights).mean()
                sigma_loss = l2_loss(student_sigmas, teacher_sigmas.detach()).mean()
                
                # Value distillation loss (MSE, no sigma weighting)
                value_loss = torch.tensor(0.0, device=self.ppo_device)
                if self.use_value_distillation and teacher_values is not None and student_values is not None:
                    value_loss = torch.nn.functional.mse_loss(student_values, teacher_values.detach())
                
                total_loss = mu_loss + sigma_loss + value_loss
            
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
            
            # Preprocess actions for environment
            if self.clip_actions:
                stepping_actions = torch.clamp(stepping_actions, -1.0, 1.0)
            stepping_actions = self.preprocess_actions(stepping_actions)
            
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
                max_ep_len = isaac_env.max_episode_length_s
                for term_name in self.reward_term_names:
                    # Initialize accumulator lists if needed
                    if term_name not in self.teacher_term_accum:
                        self.teacher_term_accum[term_name] = []
                    if term_name not in self.student_term_accum:
                        self.student_term_accum[term_name] = []
                    
                    # Teacher - accumulate instead of immediate log
                    if len(teacher_done_indices) > 0:
                        teacher_sums = self.teacher_episode_sums[term_name][teacher_done_indices]
                        val = (teacher_sums.mean() / max_ep_len).item()
                        self.teacher_term_accum[term_name].append(val)
                        # Reset for done envs
                        self.teacher_episode_sums[term_name][teacher_done_indices] = 0
                    
                    # Student - accumulate instead of immediate log
                    if len(student_done_indices) > 0:
                        student_sums = self.student_episode_sums[term_name][student_done_indices]
                        val = (student_sums.mean() / max_ep_len).item()
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
                self.writer.add_scalar('losses/value_loss', value_loss.item(), frame)
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
