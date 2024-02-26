import numpy as np
import torch
from pfrl.agents import PPO

from agents.nets import get_net
from agents.models.calculate_output_size import conv2d_size_out
from agents.agent import IndependentAgent, Agent
from agents.utils import set_pfrl_agent_mode, AGENT_MODES


class IPPO(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number, **kwargs):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            
            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])

            model = get_net(agent=self.__class__.__name__, 
                            net=kwargs.get("net"), 
                            activation=kwargs.get("activation"), 
                            obs_space=obs_space, 
                            act_space=act_space, 
                            h=h, 
                            w=w, 
                            negative_slope=kwargs.get("negative_slope")
            )()

            self.agents[key] = PFRLPPOAgent(config, obs_space, act_space, model)
            if self.config['load']:
                print('LOADING SAVED MODEL FOR EVALUATION')
                self.agents[key].load(config["models_for_visualization"] + key + ".pt")
                self.agents[key].agent.training = False


class PFRLPPOAgent(Agent):
    def __init__(self, config, obs_space, act_space, model):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4, eps=1e-5)
        self.agent = PPO(self.model, self.optimizer, gpu=self.device.index,
                         phi=lambda x: np.asarray(x, dtype=np.float32),
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         update_interval=1024,
                         minibatch_size=256,
                         epochs=4,
                         standardize_advantages=True,
                         entropy_coef=0.001,
                         max_grad_norm=0.5)

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path+'.pt')

    def load(self, path):
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        self.optimizer.load_state_dict(torch.load(path)['optimizer_state_dict'])

    def set_mode(self, mode: AGENT_MODES):
        set_pfrl_agent_mode(self.agent, mode)
