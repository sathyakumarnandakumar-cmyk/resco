from typing import Literal
from pfrl.agent import Agent as PFRLAgent

AGENT_MODES = Literal["training", "validation"]


def set_pfrl_agent_mode(agent: PFRLAgent, mode: AGENT_MODES):
    if mode == "training":
        training = True
    elif mode == "validation":
        training = False
    else:
        raise ValueError(f"Unsupported agent mode: {mode}")

    agent.training = training
