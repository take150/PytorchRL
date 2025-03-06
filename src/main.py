from logging import getLogger, basicConfig, INFO

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN on a specified task")
    parser.add_argument('--task', type=str, default='CartPole', help='Task to run')
    parser.add_argument('--task_version', type=str, default='-v1', help='Version to run')
    parser.add_argument('--algorithm', type=str, default='DQN', help='Algorithm to use')
    return parser.parse_args()

args = parse_args()
task = args.task
task_version = args.task_version
algorithm = args.algorithm
algorithm_lower = algorithm.lower()

import sys
import os
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gymnasium as gym
import importlib

agent_module = importlib.import_module(f"src.algorithms.{algorithm_lower}.agent")
Agent = getattr(agent_module, "Agent")
    
@hydra.main(config_path="config", config_name=task + '_' + algorithm, version_base=None)
def main(cfg: DictConfig):
    env = gym.make(task+task_version)
    agent = Agent(cfg=cfg)

    timesteps = 0
    while timesteps < cfg['total_timesteps']:
        done = False
        state, _ = env.reset()
        steps = 0
        while steps < cfg['max_step']:
            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.add_experience(state, action, reward, next_state, done)
            agent.update()
            state = next_state

            timesteps += 1
            steps += 1

            agent.set_epsilon()

            if done:
                break
            if timesteps % agent.target_update == 0:
                agent.sync_net()
            
            if timesteps % cfg['log_steps'] == 0:
                logger.info("time_steps: %d", timesteps)
    agent.save_model()
    env.close()

    env = gym.make(task+task_version, render_mode="human")
    done = False
    state, _  = env.reset()
    step = 0
    agent.epsilon = 0.0
    while step < cfg['max_step']:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        if done:
            break
        
            
if __name__ == '__main__':
    main()