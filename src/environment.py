from src.model import APILLM
import pandas as pd

@dataclass
class EnvironmentConfig:
    csv_path: str = "data.csv"
    model: str = "grok-4.1-fast"
    pipeline_mode: str = "explore"
    max_turns: int = 10
    target_questions: int = 10

class Environment:
    def __init__(
        self,
        csv_path: str = "data.csv",
        config: EnvironmentConfig = EnvironmentConfig(),
        ):
        self.model = APILLM(model=config.model)
        self.pipeline_mode = config.pipeline_mode
        self.max_turns = config.max_turns
        self.target_questions = config.target_questions
        self.is_completed = False
        self.state = None

    def init_state(self, ):
        pass

    def setup_state(self, state):
        pass

    def get_prompt_message(self, state):
        pass

    def act(self, state, observation):
        pass

    def update_state(self, state, action):
        pass

    def rollout(self):
        state = self.init_state(input, self.model, sampling_args)
        state = self.setup_state(state)
        while not self.is_comleted(state):
            prompt_message = self.get_prompt_message(state)
            response = self.get_model_response(prompt_message)
            self.add_model_response(state, response)
        return state
