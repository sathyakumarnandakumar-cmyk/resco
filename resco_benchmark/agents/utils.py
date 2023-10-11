from typing import Literal

def set_pfrl_agent_mode(agent, mode: Literal['train', 'eval']):
    if mode == 'train':
        training = True
    elif mode == 'eval':
        training = False
    else:
        raise ValueError(f"Unsupported agent mode: {mode}")
    
    agent.training = training