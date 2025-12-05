#!/usr/bin/env python3
"""
Video recorder for evaluating and recording the best checkpoint.
Captures video of agent performance using gymnasium's video recording.
"""


import os
# Ensure deterministic CuBLAS workspace for reproducibility when
# `torch.use_deterministic_algorithms(True)` is enabled. This must be set
# before any CUDA/CuBLAS initialization. Set default if not provided.
if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import numpy as np
import torch
import logging
import re
import glob
import json
try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
except ImportError:
    import gym
    from gym.wrappers import Monitor as RecordVideo

from dinov2_transformerxl_ppo import PPOTrainer


def record_checkpoint_video(
    trainer: PPOTrainer,
    ckpt_path: str,
    env_id: str,
    output_dir: str = "videos",
    num_episodes: int = 20,
    deterministic: bool = True,
    max_steps_per_episode: int = 5000,
    video_prefix: str = "agent",
):
    """
    Record video of agent performance from a checkpoint.
    
    Args:
        trainer: PPOTrainer instance
        ckpt_path: Path to checkpoint file
        env_id: Gymnasium environment ID
        output_dir: Directory to save videos
        num_episodes: Number of episodes to record
        deterministic: Use deterministic actions
        max_steps_per_episode: Max steps per episode
        video_prefix: Prefix for video filenames
    """
    # Load checkpoint
    logging.info(f"Loading checkpoint: {ckpt_path}")
    trainer.load_checkpoint(ckpt_path)
    trainer.backbone.eval()
    trainer.projector.eval()
    trainer.policy.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with video recording
    # Record every episode
    env = None
    try:
        # Gymnasium style
        env = gym.make(env_id, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=output_dir,
            episode_trigger=lambda episode_id: True,  # Record all episodes
            name_prefix=video_prefix,
        )
    except TypeError:
        # Older gym style
        env = gym.make(env_id)
        env = RecordVideo(
            env,
            directory=output_dir,
            video_callable=lambda episode_id: True,
            name_prefix=video_prefix,
        )
    except Exception as e:
        logging.error(f"Failed to create environment or video recorder: {e}")
        if env is not None:
            env.close()
        raise
    
    returns = []
    lengths = []
    
    # Initialize memory
    try:
        memory = trainer.memory[:, 0:1, :].to(trainer.device).clone()
    except Exception as e:
        logging.warning(f"Could not clone trainer memory: {e}")
        memory = None
    
    logging.info(f"Recording {num_episodes} episodes... (Mode: {'Deterministic' if deterministic else 'Stochastic'})")
    
    for ep in range(num_episodes):
        try:
            obs_reset = env.reset()
            if isinstance(obs_reset, tuple) and len(obs_reset) >= 1:
                obs = obs_reset[0]
            else:
                obs = obs_reset
            done = False
            total_reward = 0.0
            steps = 0
            mem = memory.clone() if memory is not None else None
            logging.info(f"Episode {ep + 1}/{num_episodes}...")
            while not done and steps < max_steps_per_episode:
                proj = trainer.project_obs(np.expand_dims(obs, 0))  # [1, D]
                with torch.no_grad():
                    policy_out, value = trainer.policy.forward_sequence(
                        seq_proj=proj.unsqueeze(0), 
                        memory=mem
                    )
                    if trainer.policy.is_discrete:
                        logits = policy_out.squeeze(0).squeeze(0)
                        if deterministic:
                            action = int(torch.argmax(logits).cpu().item())
                        else:
                            probs = torch.softmax(logits, dim=-1)
                            action = int(torch.multinomial(probs, 1).cpu().item())
                    else:
                        mean, std = policy_out
                        mean = mean.squeeze(0).squeeze(0).cpu().numpy()
                        if deterministic:
                            action = mean
                        else:
                            std_np = std.squeeze(0).squeeze(0).cpu().numpy()
                            action = np.random.normal(mean, std_np)
                step_ret = env.step(action)
                if isinstance(step_ret, tuple) and len(step_ret) == 5:
                    next_obs, reward, terminated, truncated, info = step_ret
                    done = bool(terminated or truncated)
                elif isinstance(step_ret, tuple) and len(step_ret) == 4:
                    next_obs, reward, done, info = step_ret
                else:
                    raise RuntimeError(f"Unexpected env.step return: {step_ret}")
                total_reward += reward
                steps += 1
                try:
                    proj_token = proj.unsqueeze(0).cpu()
                    if mem is not None:
                        mem = trainer.policy.update_memory(mem, proj_token.to(trainer.device))
                except Exception as e:
                    logging.debug(f"Memory update failed: {e}")
                obs = next_obs
            returns.append(total_reward)
            lengths.append(steps)
            logging.info(f"Return: {total_reward:.2f}, Steps: {steps}")
        except Exception as e:
            logging.error(f"Error in episode {ep+1}: {e}")
            continue
    
    if env is not None:
        env.close()
    
    # Print summary statistics
    logging.info("\n" + "="*50)
    logging.info("VIDEO RECORDING SUMMARY")
    logging.info("="*50)
    logging.info(f"Checkpoint: {os.path.basename(ckpt_path)}")
    logging.info(f"Environment: {env_id}")
    logging.info(f"Episodes recorded: {num_episodes}")
    logging.info(f"Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    logging.info(f"\nPerformance Statistics:")
    logging.info(f"  Mean Return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    logging.info(f"  Median Return: {np.median(returns):.2f}")
    logging.info(f"  Min/Max Return: {np.min(returns):.2f} / {np.max(returns):.2f}")
    logging.info(f"  Mean Length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")
    logging.info(f"\nVideos saved to: {output_dir}/")
    logging.info("="*50)
    
    return returns, lengths


def main():
    parser = argparse.ArgumentParser(
        description="Record video of agent performance from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_lunarlander/ckpt_1638400.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="LunarLander-v3",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to record",
    )
    
    ### --- MODIFIED SECTION --- ###
    # Changed "--stochastic" to "--deterministic"
    # The default is now STOCHASTIC (deterministic=False)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (argmax/mean) instead of stochastic sampling",
    )
    ### --- END MODIFIED SECTION --- ###
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed for reproducibility",
    )
    parser.add_argument(
        "--video_prefix",
        type=str,
        default="agent",
        help="Prefix for video filenames",
    )
    
    args, unknown = parser.parse_known_args()
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    # Helper: discover best checkpoint produced by eval_checkpoint.py
    def discover_best_checkpoint():
        # 1) try eval_summary_*.json (latest)
        try:
            summaries = glob.glob("eval_summary_*.json")
            if summaries:
                summaries.sort(key=os.path.getmtime)
                latest = summaries[-1]
                with open(latest, 'r') as f:
                    data = json.load(f)
                results = data.get('results') or data.get('results_sorted') or []
                if isinstance(results, list) and len(results) > 0:
                    first = results[0]
                    if isinstance(first, dict) and 'ckpt' in first:
                        return first['ckpt']
        except Exception as e:
            logging.debug(f"Could not read eval_summary json: {e}")
        # 2) try quick_eval_best.py
        try:
            if os.path.exists('quick_eval_best.py'):
                with open('quick_eval_best.py', 'r') as f:
                    content = f.read()
                m = re.search(r'CKPT\s*=\s*"([^"]+)"', content)
                if m:
                    return m.group(1)
        except Exception as e:
            logging.debug(f"Could not read quick_eval_best.py: {e}")
        return None
    # If user requested the special checkpoint name "BEST", try to discover it
    if isinstance(args.checkpoint, str) and args.checkpoint.strip().upper() == "BEST":
        discovered = discover_best_checkpoint()
        if discovered:
            logging.info(f"Discovered best checkpoint: {discovered}")
            args.checkpoint = discovered
        else:
            logging.error("Requested BEST checkpoint but none found (no eval_summary_*.json or quick_eval_best.py). Exiting.")
            return 1

    # Create trainer instance
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    logging.info(f"Environment: {args.env}")
    
    ### --- MODIFIED LINE --- ###
    # Pass render_mode="rgb_array" to the trainer's constructor
    # This ensures its internal env-making logic matches
    trainer = PPOTrainer(
        env_id=args.env,
        seed=args.seed,
        num_envs=1,
        device=device,
        total_steps=1,  # Not training, just evaluating
        freeze_backbone=True,
        render_mode="rgb_array" # <-- ADDED FOR CONSISTENCY
    )
    
    # Record videos
    try:
        ### --- MODIFIED LINE --- ###
        # Pass the new 'args.deterministic' flag
        record_checkpoint_video(
            trainer=trainer,
            ckpt_path=args.checkpoint,
            env_id=args.env,
            output_dir=args.output_dir,
            num_episodes=args.episodes,
            deterministic=args.deterministic, # <-- CHANGED from 'not args.stochastic'
            max_steps_per_episode=args.max_steps,
            video_prefix=args.video_prefix,
        )
    except Exception as e:
        logging.error(f"Error during recording: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())