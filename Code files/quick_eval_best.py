# quick_eval_best.py

import os
# Ensure deterministic CuBLAS workspace for reproducibility when
# `torch.use_deterministic_algorithms(True)` is enabled. This must be set
# before any CUDA/CuBLAS initialization. Set default if not provided.
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import logging
import numpy as np
import torch

try:
    import gymnasium as gym
except Exception:
    import gym


### --- MODIFIED SECTION --- ###

# Set this to your checkpoint path
CKPT = "checkpoints_lunarlander/ckpt_1638400.pth" 
# Set this to the correct environment
ENV = "LunarLander-v3"
EPISODES = 10

# --- NEW TOGGLE ---
# Set to False to sample actions (stochastic), which matches your training.
# Set to True to use deterministic actions (argmax/mean).
DETERMINISTIC = False

### --- END MODIFIED SECTION --- ###


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# reproducible seed
SEED = 0
logging.info(f"Loading checkpoint: {CKPT}")
logging.info(f"Evaluating env: {ENV} for {EPISODES} episodes")
logging.info(f"Using {'DETERMINISTIC' if DETERMINISTIC else 'STOCHASTIC'} actions")

trainer = PPOTrainer(env_id=ENV, seed=SEED, num_envs=1, device=device, total_steps=1, freeze_backbone=True, render_mode="rgb_array")
trainer.load_checkpoint(CKPT)
trainer.backbone.eval(); trainer.projector.eval(); trainer.policy.eval()

returns = []
for ep in range(EPISODES):
    env = None
    try:
        ### --- MODIFIED LINE --- ###
        # Ensure we get 'rgb_array' for the ViT backbone
        env = gym.make(ENV, render_mode="rgb_array")
        
        obs_reset = env.reset()
        if isinstance(obs_reset, tuple) and len(obs_reset) >= 1:
            obs = obs_reset[0]
        else:
            obs = obs_reset
        done = False
        R = 0.0
        steps = 0
        memory = trainer.memory[:, 0:1, :].to(trainer.device).clone() if hasattr(trainer, "memory") else None
        
        while not done and steps < 5000:
            proj = trainer.project_obs(np.expand_dims(obs, 0))  # [1,D]
            with torch.no_grad():
                policy_out, _ = trainer.policy.forward_sequence(seq_proj=proj.unsqueeze(0), memory=memory)
                
                ### --- MODIFIED ACTION SELECTION --- ###
                if trainer.policy.is_discrete:
                    logits = policy_out.squeeze(0).squeeze(0)
                    if DETERMINISTIC:
                        action = int(torch.argmax(logits).cpu().item())
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        action = int(torch.multinomial(probs, 1).cpu().item())
                else:
                    # Handle continuous case
                    mean, std = policy_out
                    mean = mean.squeeze(0).squeeze(0).cpu().numpy()
                    if DETERMINISTIC:
                        action = mean
                    else:
                        std_np = std.squeeze(0).squeeze(0).cpu().numpy()
                        action = np.random.normal(mean, std_np)
                ### --- END MODIFIED ACTION SELECTION --- ###

            step_ret = env.step(action)
            if isinstance(step_ret, tuple) and len(step_ret) == 5:
                next_obs, r, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                next_obs, r, done, info = step_ret
            else:
                raise RuntimeError(f"Unexpected env.step return: {step_ret}")
            R += r
            steps += 1
            try:
                proj_token = proj.unsqueeze(0).cpu()
                if memory is not None:
                    memory = trainer.policy.update_memory(memory, proj_token.to(device))
            except Exception as e:
                logging.debug(f"Memory update failed: {e}")
            obs = next_obs
        logging.info(f"Episode {ep+1}/{EPISODES}: return={R:.2f}, steps={steps}")
        returns.append(R)
    except Exception as e:
        logging.error(f"Error in episode {ep}: {e}")
    finally:
        if env is not None:
            env.close()

logging.info("\n" + "="*30)
logging.info(f"EVALUATION COMPLETE ({'DETERMINISTIC' if DETERMINISTIC else 'STOCHASTIC'})")
logging.info(f"  Checkpoint: {CKPT}")
logging.info(f"  Mean Return: {np.mean(returns):.2f}")
logging.info(f"  Std Dev: {np.std(returns):.2f}")
logging.info(f"  Min/Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
logging.info("="*30)