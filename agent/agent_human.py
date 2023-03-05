from agent.agent import Agent
import pygame


class AgentHuman(Agent):
    # Two modes:
    # Real-time: For environments that are real-time, the human agent can respond to the environment at any point,
    # but the environment will not wait for the human agent to respond
    # Turn-based: For environments that are turn-based, the human agent can respond to the environment at their own pace

    def __init__(self, name: str, key_to_action_mapping: dict, real_time: bool = False):
        super().__init__(name)
        self.real_time = real_time
        self.key_to_action_mapping = key_to_action_mapping

    def get_action(self, obs, action_mask, training=False):
        # If real-time, then get the keys that are pressed and return the action
        if self.real_time:
            # Get the keys pressed using pygame
            try:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        # Get the action from the key
                        # TODO: Add support for combinations of actions, and maybe continuous actions
                        # Currently only the first key listed in the key_to_action_mapping that is pressed will be used
                        for key in self.key_to_action_mapping:
                            if event.key == key:
                                return self.key_to_action_mapping[key]
            except pygame.error:
                pass
            # If no key is pressed, return the first action in key_to_action_mapping
            return list(self.key_to_action_mapping.values())[0]
        else:  # If turn-based, then wait for the user to enter an action into the console
            while True:
                valid_actions = [i for i in range(
                    len(action_mask)) if action_mask[i] == 1]
                print(f"Valid actions: {valid_actions}")
                action = input("Enter an valid action: ")
                try:
                    action = int(action)
                    if action in valid_actions:
                        return action  # Actions should be a non-negative integer, without gaps
                except ValueError:
                    pass

    def reset(self):
        pass

    def save(self, filename=None):
        pass

    def load(self, filename=None):
        pass

    def __str__(self):
        return "Human"
