"""Verifiers environment entry point for Prime-RL.

Verifiers resolves the environment id ``csv-agent`` to this import module.
The implementation stays in ``src.training`` so csv-agent's canonical episode
adapter can be tested and reused without duplicating training code.
"""

from src.training.rl_env import CSVAgentRLEnv, episodes_to_dataset, load_environment

__all__ = ["CSVAgentRLEnv", "episodes_to_dataset", "load_environment"]
