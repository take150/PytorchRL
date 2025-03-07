from logging import getLogger, basicConfig, INFO

basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)

import sys
import os
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gymnasium as gym
from src.algorithms.dqn.agent import Agent
    
@hydra.main(config_path="config", config_name="cartpole_dqn", version_base=None)
def main(cfg: DictConfig):
    env = gym.make("CartPole-v1")
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

    env = gym.make("CartPole-v1", render_mode="human")
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