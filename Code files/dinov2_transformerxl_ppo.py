import os
# Ensure deterministic CuBLAS workspace for reproducibility when
# `torch.use_deterministic_algorithms(True)` is enabled. This must be set
# before any CUDA/CuBLAS initialization (i.e. before importing torch or
# creating CUDA tensors). Use ':4096:8' or ':16:8' as recommended by NVIDIA.
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import math
import time
from typing import Optional, Tuple, List

import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv

# Utilities: vector env factory

def make_env_fn(env_id: str, seed: int = 0, render_mode: Optional[str]=None):
    def _thunk():
        # pass render_mode to gym.make when provided (needed to get image observations)
        if render_mode is not None:
            env = gym.make(env_id, render_mode=render_mode)
        else:
            env = gym.make(env_id)
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                pass
        return env
    return _thunk


# convenience: create SyncVectorEnv from gymnasium
def make_vector_envs(env_id: str, num_envs: int, base_seed: int = 0, render_mode: Optional[str]=None):
    fns = [make_env_fn(env_id, seed=base_seed + i, render_mode=render_mode) for i in range(num_envs)]
    return SyncVectorEnv(fns)


def set_global_seed(seed: int):
    """Set seeds for python, numpy, torch and configure deterministic behavior.

    Note: full determinism across different hardware and library versions is not guaranteed,
    but this sets the recommended flags to maximize reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch deterministic settings
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_vit_backbone(backbone_name: str = "vit_b_16", device: Optional[torch.device] = None, dino_ckpt_path: Optional[str] = None, obs_space=None):
   
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if the observation space is not image-shaped, return an identity backbone
    if obs_space is not None:
        shape = getattr(obs_space, 'shape', None)
        if shape is not None and len(shape) != 3:
            # vector observation
            class IdentityBackbone(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.dim = dim
                def forward(self, x):
                    return x
            feat_dim = int(np.prod(shape)) if shape is not None else 1
            return IdentityBackbone(feat_dim), feat_dim

    if backbone_name == "dino_v2" and dino_ckpt_path is not None:
        # TODO: implement actual DINO-v2 checkpoint loading depending on format.
        raise NotImplementedError("Please replace loader with your DINO-v2 checkpoint loading code and mapping.")
    else:
        # fallback: torchvision ViT base 16
        vit = models.vit_b_16(pretrained=True)
        feat_dim = vit.heads.head.in_features if hasattr(vit, "heads") and hasattr(vit.heads, "head") else vit.hidden_dim
        vit.heads = nn.Identity()
        vit.to(device)
        vit.eval()
        return vit, feat_dim


def default_image_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # DINO / ViT default
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class Projector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        # x: [B, in_dim]
        return self.net(x)

# ----------------------------
# Simple Transformer-XL style recurrent module
# - We'll implement a causal Transformer encoder and support concatenating memory (memory_len tokens)
# - Memory is a tensor [memory_len, batch, dim] for each batch
# - We represent each time-step's state as a single token (per-frame embedding)
# ----------------------------
def causal_attention_mask(sz_q: int, sz_k: int, device):
    # returns mask shape (sz_q, sz_k) True where masked
    # we want to allow attending only to <= current position (causal)
    idxs_q = torch.arange(sz_q, device=device)[:, None]
    idxs_k = torch.arange(sz_k, device=device)[None, :]
    mask = idxs_k > idxs_q  # True where k > q (disallowed)
    return mask  # (sz_q, sz_k)

class SmallCausalTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=False)  # we'll use seq_len x batch x d
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # x: seq_len x batch x d_model
        # transformer expects src_mask shape (seq_len, seq_len) with True where masked
        return self.transformer(x, mask=attn_mask)


# Policy: takes sequences of projected embeddings + recurrent memory -> action logits/means + value

class RecurrentPolicy(nn.Module):
    def __init__(
        self,
        proj_dim: int,
        action_space,
        transformer_d_model: int = 512,
        transformer_heads: int = 8,
        transformer_layers: int = 2,
        memory_segments: int = 4,
        segment_length: int = 16,
        compress_ratio: int = 1,  # compression for old segments (1 = no compression)
        device: Optional[torch.device] = None,
        fine_tune_backbone: bool = False,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.proj_dim = proj_dim
        self.d_model = transformer_d_model
        # a linear to map proj_dim to transformer d_model
        self.input_mapper = nn.Linear(proj_dim, transformer_d_model)
        self.transformer = SmallCausalTransformer(d_model=transformer_d_model,
                                                  nhead=transformer_heads,
                                                  num_layers=transformer_layers)
        # heads
        if isinstance(action_space, gym.spaces.Discrete):
            self.is_discrete = True
            self.action_dim = action_space.n
            self.policy_head = nn.Linear(transformer_d_model, self.action_dim)
        elif isinstance(action_space, gym.spaces.Box):
            self.is_discrete = False
            self.action_dim = int(np.prod(action_space.shape))
            self.policy_head = nn.Linear(transformer_d_model, self.action_dim)
            self.logstd = nn.Parameter(torch.zeros(self.action_dim, device=self.device))
        else:
            raise NotImplementedError("Action space not supported")

        self.value_head = nn.Linear(transformer_d_model, 1)
        self.memory_segments = memory_segments
        self.segment_length = segment_length
        self.compress_ratio = max(1, compress_ratio)

        # initialize memory buffers externally per env (see trainer)
        # move to device
        self.to(self.device)

    def forward_sequence(self, seq_proj: torch.Tensor, memory: Optional[torch.Tensor] = None):

        T, N, _ = seq_proj.shape
        x = self.input_mapper(seq_proj)  # [T, N, d_model]
        if memory is not None:
            # map memory to d_model and concat (memory first, then current sequence)
            mem_mapped = self.input_mapper(memory)  # [M, N, d_model]
            cat = torch.cat([mem_mapped, x], dim=0)  # [M+T, N, d_model]
            seq = cat
            mem_len = mem_mapped.shape[0]
        else:
            seq = x
            mem_len = 0

        # build causal mask so positions can attend to previous positions (including memory)
        seq_len = seq.shape[0]
        mask = causal_attention_mask(seq_len, seq_len, device=seq.device)  # True masked
        out = self.transformer(seq, attn_mask=mask)  # [seq_len, N, d_model]
        # take only the last T outputs (corresponding to current sequence)
        out_current = out[mem_len:]  # [T, N, d_model]
        # compute policy & value per timestep
        if self.is_discrete:
            logits = self.policy_head(out_current)  # [T, N, action_dim]
            values = self.value_head(out_current).squeeze(-1)  # [T, N]
            return logits, values
        else:
            mean = self.policy_head(out_current)
            std = torch.exp(self.logstd).unsqueeze(0).expand_as(mean)
            values = self.value_head(out_current).squeeze(-1)
            return (mean, std), values

    def init_memory(self, num_envs: int, device: Optional[torch.device] = None):
        device = device or self.device
        # memory represented in proj_dim space (same space as projector output)
        # We'll store memory in proj_dim space for easier compression / storage.
        mem_slots = self.memory_segments * self.segment_length // self.compress_ratio
        # shape [M, N, proj_dim]
        return torch.zeros((mem_slots, num_envs, self.proj_dim), device=device)

    def update_memory(self, memory: torch.Tensor, new_seq_proj: torch.Tensor):

        M, N, D = memory.shape
        T = new_seq_proj.shape[0]
        # flatten memory timeline -> last K tokens
        combined = torch.cat([memory, new_seq_proj], dim=0)  # [M+T, N, D]
        # keep last M tokens
        new_mem = combined[-M:]
        return new_mem.detach()  # detach to avoid backprop through old memory


# Helper: utilities for image batches processing

def preprocess_observations(obs_batch: np.ndarray, transform):

    # support uint8 inputs and both image and vector observations
    N = obs_batch.shape[0]
    # if channels last image HWC
    if transform is not None and obs_batch.ndim == 4 and obs_batch.shape[-1] in (1, 3):
        imgs = []
        for i in range(N):
            img = obs_batch[i]
            img_t = transform(img)  # returns C,H,W tensor
            imgs.append(img_t)
        imgs = torch.stack(imgs, dim=0)
        return imgs
    else:
        # vector observations or already CHW tensors
        if obs_batch.dtype == np.uint8:
            obs_batch = obs_batch.astype(np.float32) / 255.0
        return torch.from_numpy(obs_batch).float()


# PPO Trainer (with memory & multi-env)

class PPOTrainer:
    def __init__(
        self,
        env_id: str,
        seed: int = 0,
        num_envs: int = 8,
        device: Optional[torch.device] = None,
        total_steps: int = 200_000,
        rollout_length: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        ppo_clip: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        backbone_name: str = "vit_b_16",
        dino_ckpt_path: Optional[str] = None,
        proj_dim: int = 256,
        transformer_d_model: int = 512,
        transformer_layers: int = 2,
        memory_segments: int = 4,
        segment_length: int = 16,
        compress_ratio: int = 1,
        freeze_backbone: bool = True,
        render_mode: Optional[str] = None,
        save_dir: str = "checkpoints",
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # global seed for reproducibility
        self.seed = seed
        set_global_seed(self.seed)
        # allow specifying render_mode so envs can return image observations when needed
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.env_id = env_id
        self.total_steps = total_steps
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.lam = lam
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # create vector envs (use provided seed as base_seed so envs are reproducible)
        self.envs = make_vector_envs(env_id, num_envs, base_seed=self.seed, render_mode=self.render_mode)
        self.obs_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

        # backbone selection (image vs vector observations)
        self.backbone, feat_dim = load_vit_backbone(backbone_name, device=self.device, dino_ckpt_path=dino_ckpt_path, obs_space=self.obs_space)
        self.backbone_name = backbone_name
        self.backbone_feature_dim = feat_dim
        self.transform = default_image_transform()

        # projector
        self.projector = Projector(in_dim=feat_dim, proj_dim=proj_dim).to(self.device)

        # policy
        self.policy = RecurrentPolicy(proj_dim=proj_dim,
                                      action_space=self.action_space,
                                      transformer_d_model=transformer_d_model,
                                      transformer_heads=8,
                                      transformer_layers=transformer_layers,
                                      memory_segments=memory_segments,
                                      segment_length=segment_length,
                                      compress_ratio=compress_ratio,
                                      device=self.device).to(self.device)

        # optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # optimizer includes policy, projector, and optionally backbone
        params = list(self.policy.parameters()) + list(self.projector.parameters())
        if not freeze_backbone:
            params += [p for p in self.backbone.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=learning_rate, eps=1e-5)

        # memory buffers per env (projector-space memory)
        self.memory = self.policy.init_memory(num_envs=self.num_envs, device=self.device)  # [M, N, proj_dim]

    def extract_backbone_features(self, imgs_tensor: torch.Tensor) -> torch.Tensor:

        # pass through ViT backbone which expects [B, C, H, W]
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone(imgs_tensor.to(self.device))
        return feats  # [B, feat_dim]

    def project_obs(self, obs_batch: np.ndarray) -> torch.Tensor:
        imgs_t = preprocess_observations(obs_batch, self.transform).to(self.device)  # [N, C, H, W]
        feats = self.extract_backbone_features(imgs_t)  # [N, feat_dim]
        proj = self.projector(feats)  # [N, proj_dim]
        return proj  # [N, proj_dim]

    def compute_gae(self, rewards, values, dones):

        T, N = rewards.shape
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        last_gae_lam = np.zeros(N)
        next_values = np.zeros(N)
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
            last_gae_lam = delta + self.gamma * self.lam * next_nonterminal * last_gae_lam
            advs[t] = last_gae_lam
            returns[t] = advs[t] + values[t]
            next_values = values[t]
        return returns, advs

    def save_checkpoint(self, step:int):
        ckpt = {
            'backbone_name': self.backbone_name,
            'backbone_state': self.backbone.state_dict(),
            'projector_state': self.projector.state_dict(),
            'policy_state': self.policy.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'memory': self.memory.cpu(),
            'step': step
        }
        path = os.path.join(self.save_dir, f"ckpt_{step}.pth")
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # load states; be mindful if architecture changed
        self.backbone.load_state_dict(ckpt['backbone_state'], strict=False)
        self.projector.load_state_dict(ckpt['projector_state'], strict=False)
        self.policy.load_state_dict(ckpt['policy_state'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.memory = ckpt['memory'].to(self.device)
        print(f"Loaded checkpoint: {path}")

    def train(self):
        obs_reset = self.envs.reset()
        # gymnasium.reset() returns (obs, infos) for vector envs; handle both forms
        if isinstance(obs_reset, tuple) and len(obs_reset) == 2:
            obs, _ = obs_reset
        else:
            obs = obs_reset
        # make sure obs is numpy array [N, ...]
        total_steps = 0
        start_time = time.time()
        ep_returns = np.zeros(self.num_envs)
        ep_lengths = np.zeros(self.num_envs)
        episode_info = []

        while total_steps < self.total_steps:
            # rollout buffers
            imgs_buf = []
            proj_buf = []  # projected embeddings per time
            actions_buf = []
            logp_buf = []
            rewards_buf = []
            dones_buf = []
            values_buf = []

            # we will collect rollout_length steps
            for step in range(self.rollout_length):
                # prepare images -> proj embeddings
                proj = self.project_obs(obs)  # [N, proj_dim]
                # reshape to [1, N, proj_dim] for sequence step
                proj_step = proj.unsqueeze(0)  # [1, N, D]
                # supply memory (proj-space) to policy
                with torch.no_grad():
                    policy_out, value = self.policy.forward_sequence(seq_proj=proj_step, memory=self.memory)
                    # policy_out: logits [1,N,A] or (mean,std) for continuous
                    value_np = value.squeeze(0).cpu().numpy()  # [N]
                    values_buf.append(value_np)
                    if self.policy.is_discrete:
                        logits = policy_out.squeeze(0)  # [N,A]
                        probs = torch.softmax(logits, dim=-1)
                        actions = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
                        logp = torch.log_softmax(logits, dim=-1).gather(1, torch.from_numpy(actions).to(self.device).unsqueeze(1)).squeeze(1).cpu().numpy()
                    else:
                        mean, std = policy_out
                        dist = torch.distributions.Normal(mean.squeeze(0), std.squeeze(0))
                        actions_t = dist.sample()
                        actions = actions_t.cpu().numpy()
                        logp = dist.log_prob(actions_t).sum(-1).cpu().numpy()

                # step envs
                step_ret = self.envs.step(actions)
                # gymnasium VectorEnv.step returns either
                # (obs, rewards, terminated, truncated, infos) or the older (obs, rewards, dones, infos)
                if isinstance(step_ret, tuple) and len(step_ret) == 5:
                    next_obs, rewards, terminated, truncated, infos = step_ret
                    dones = np.logical_or(terminated, truncated)
                elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                    next_obs, rewards, dones, infos = step_ret
                else:
                    raise RuntimeError(f"Unexpected return from envs.step: len={len(step_ret) if isinstance(step_ret, tuple) else 'unknown'}")
                imgs_buf.append(obs)  # store raw obs for potential diagnostics
                proj_buf.append(proj.detach().cpu().numpy())  # store proj (for memory updates after rollout)
                actions_buf.append(actions)
                logp_buf.append(logp)
                rewards_buf.append(rewards)
                dones_buf.append(dones)
                values_buf[-1] = value_np  # ensured

                # bookkeeping
                ep_returns += rewards
                ep_lengths += 1
                for i, d in enumerate(dones):
                    if d:
                        episode_info.append({"reward": float(ep_returns[i]), "length": int(ep_lengths[i])})
                        ep_returns[i] = 0
                        ep_lengths[i] = 0

                obs = next_obs
                total_steps += self.num_envs

            # convert buffers -> arrays for PPO updates
            # shapes: [T, N, ...]
            rewards_arr = np.stack(rewards_buf, axis=0)
            dones_arr = np.stack(dones_buf, axis=0).astype(np.float32)
            values_arr = np.stack(values_buf, axis=0)
            actions_arr = np.stack(actions_buf, axis=0)
            logp_arr = np.stack(logp_buf, axis=0)
            proj_arr = np.stack(proj_buf, axis=0)  # [T, N, proj_dim]

            # compute returns and advantages
            returns, advs = self.compute_gae(rewards_arr, values_arr, dones_arr)
            # flatten for optimization
            T, N = returns.shape
            flat_returns = returns.reshape(T * N)
            flat_advs = advs.reshape(T * N)
            flat_actions = actions_arr.reshape(T * N, -1) if (not self.policy.is_discrete) else actions_arr.reshape(T * N)
            flat_logp_old = logp_arr.reshape(T * N)
            flat_proj = proj_arr.reshape(T * N, -1)  # flatten

            # We'll perform PPO updates using minibatches sampled from T*N examples.
            # To compute current logp & values we process in minibatches by reconstructing sequences
            # For simplicity we'll process per timestep (no long sequences for transformer during updates), but keep memory usage okay.
            # NOTE: this is an approximation — for precise segment-level recurrence training you'd run sequences through the transformer with the same memory as used during collection.
            # We'll use the memory as it stands (self.memory) and process each timestep independently by feeding memory + single timestep.
            # Prepare dataset indices
            batch_size = T * N
            idxs = np.arange(batch_size)
            for epoch in range(self.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, batch_size, self.minibatch_size):
                    mb_idx = idxs[start:start + self.minibatch_size]
                    # convert flat indices -> (t, n)
                    t_idx = (mb_idx // N).astype(int)
                    n_idx = (mb_idx % N).astype(int)
                    # gather proj tokens for this minibatch
                    mb_proj = torch.from_numpy(flat_proj[mb_idx]).to(self.device).float()  # [B, proj_dim]
                    # reshape to seq_len 1 for transformer's forward
                    mb_seq = mb_proj.unsqueeze(0)  # [1, B, proj_dim]
                    # create memory for each minibatch element by indexing self.memory along env dim
                    # memory: [M, N, D] -> we need [M, B, D]
                    # gather along env dimension
                    mem = self.memory[:, n_idx, :].to(self.device)  # [M, B, D]
                    # forward
                    policy_out, value_pred = self.policy.forward_sequence(seq_proj=mb_seq, memory=mem)
                    # policy_out -> [1, B, A] etc.
                    if self.policy.is_discrete:
                        logits = policy_out.squeeze(0)  # [B, A]
                        logp_all = torch.log_softmax(logits, dim=-1)
                        # gather action logp
                        mb_actions = torch.from_numpy(flat_actions[mb_idx]).to(self.device).long()
                        logp = logp_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)
                        entropy = -(torch.exp(logp_all) * logp_all).sum(-1).mean()
                    else:
                        mean, std = policy_out
                        mean = mean.squeeze(0)
                        std = std.squeeze(0)
                        dist = torch.distributions.Normal(mean, std)
                        mb_actions = torch.from_numpy(flat_actions[mb_idx]).to(self.device).float()
                        logp = dist.log_prob(mb_actions).sum(-1)
                        entropy = dist.entropy().sum(-1).mean()

                    value_pred = value_pred.squeeze(0)  # [B]
                    # compute advantages & returns for minibatch
                    mb_returns = torch.from_numpy(flat_returns[mb_idx]).to(self.device).float()
                    mb_advs = torch.from_numpy(flat_advs[mb_idx]).to(self.device).float()
                    mb_logp_old = torch.from_numpy(flat_logp_old[mb_idx]).to(self.device).float()

                    ratio = torch.exp(logp - mb_logp_old)
                    surr1 = ratio * mb_advs
                    surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_advs
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (mb_returns - value_pred).pow(2).mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()

            # After optimization, update memory by appending last rollout's proj tokens (we append per env)
            # proj_arr: [T, N, D]
            proj_torch = torch.from_numpy(proj_arr).to(self.device).float()  # [T, N, D]
            # append sequentially: for each env, update memory with last segment_length tokens; here we update memory globally by using the T last tokens.
            # Reshape proj_torch into [T, N, D] and update memory using policy.update_memory
            # For simplicity, we call update_memory once with the last segment_length steps
            last_segment = proj_torch[-self.policy.segment_length:] if proj_torch.shape[0] >= self.policy.segment_length else proj_torch
            # last_segment: [t', N, D] where t' <= segment_length
            # pad if needed to segment_length by repeating last
            if last_segment.shape[0] < self.policy.segment_length:
                pad = self.policy.segment_length - last_segment.shape[0]
                pad_t = last_segment[-1:].expand(pad, -1, -1)
                last_segment = torch.cat([last_segment, pad_t], dim=0)
            # update memory
            self.memory = self.policy.update_memory(self.memory, last_segment.cpu().to(self.device))

            # logging
            elapsed = time.time() - start_time
            avg_return = np.mean([e['reward'] for e in episode_info[-50:]]) if len(episode_info) > 0 else 0.0
            print(f"Steps: {total_steps}/{self.total_steps}, elapsed: {elapsed:.1f}s, recent_avg_return: {avg_return:.2f}")

            # save checkpoint periodically
            if total_steps % (self.rollout_length * self.num_envs * 10) == 0:
                self.save_checkpoint(total_steps)

        # final save
        self.save_checkpoint(total_steps)
        self.envs.close()


# Main: run trainer with defaults

# if __name__ == "__main__":
#     # CHANGE THESE to match your environment & compute
#     ENV_ID = "CartPole-v1"  # replace with visual env (should output images)
#     NUM_ENVS = 8
#     TOTAL_STEPS = 50_000  # for quick test; increase for real training
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # global seed for reproducibility across runs/GPUs
#     SEED = 0

#     trainer = PPOTrainer(
#         env_id=ENV_ID,
#         seed=SEED,
#         num_envs=NUM_ENVS,
#         device=device,
#         total_steps=TOTAL_STEPS,
#         rollout_length=64,
#         learning_rate=2.5e-4,
#         gamma=0.99,
#         lam=0.95,
#         ppo_clip=0.2,
#         ppo_epochs=4,
#         minibatch_size=256,
#         backbone_name="vit_b_16",  # swap to "dino_v2" and provide dino_ckpt_path for DINO
#         dino_ckpt_path=None,
#         proj_dim=256,
#         transformer_d_model=512,
#         transformer_layers=2,
#         memory_segments=4,
#         segment_length=16,
#         compress_ratio=1,
#         freeze_backbone=True,
#         save_dir="checkpoints"
#   )

# [NEW Recommended __main__ block for LunarLander-v3]

if __name__ == "__main__":
    # === Tuned for LunarLander-v3 (Visual) ===
    ENV_ID = "LunarLander-v3"
    NUM_ENVS = 8
    TOTAL_STEPS = 2_000_000  # Much harder env, needs more steps!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 0
    # CRITICAL: We MUST tell the env to give us pixels!
    RENDER_MODE = "rgb_array" 

    trainer = PPOTrainer(
        env_id=ENV_ID,
        seed=SEED,
        num_envs=NUM_ENVS,
        device=device,
        total_steps=TOTAL_STEPS,
        rollout_length=2048,      # <-- INCREASED: Needs long horizon for sparse reward
        learning_rate=1e-4,       # <-- DECREASED: More stable for harder task
        gamma=0.99,               # <-- Stays high (good for sparse reward)
        lam=0.95,
        ppo_clip=0.2,
        ppo_epochs=10,            # <-- INCREASED: Learn more from each rollout
        minibatch_size=256,       # <-- OK: (2048*8) / 256 = 64 minibatches
        backbone_name="vit_b_16",
        dino_ckpt_path=None,
        proj_dim=256,
        transformer_d_model=512,
        transformer_layers=2,
        memory_segments=4,
        segment_length=16,
        compress_ratio=1,
        freeze_backbone=True,
        render_mode=RENDER_MODE,  # <-- ADDED: Pass this to the env!
        save_dir="checkpoints_lunarlander" # <-- Optional: new save dir
    )

    trainer.train()
