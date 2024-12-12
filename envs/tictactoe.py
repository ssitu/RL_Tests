from envs.env import Env
from agent.agent import Agent
import numpy as np
from typing import Tuple
from time import sleep


class TicTacToe(Env):

    def __init__(self, enemy_agent: Agent, human_render=False, human_render_delay=2, reload_enemy_interval=100):
        super().__init__(human_render)
        # Initialize the board as a 3x3 numpy array of zeros
        self.board = np.zeros((3, 3), dtype=int)
        # The id of the agent that is playing against the enemy, 1 or 2
        self.agent_id = 2  # Will set to 1 on reset
        # Id for the enemy agent, 1 or 2, but not the same as agent_player
        self.enemy_id = 1  # Will set to 2 on reset
        self.enemy_agent = enemy_agent
        # The number of games played
        self.games_played = -1 # Will be set to 0 on reset

        # How many games in between reloading the enemy agent
        self.reload_enemy_interval = reload_enemy_interval

        # Seconds between moves when human_render is True
        self.human_render_delay = human_render_delay

    @classmethod
    def get_observation_space(cls) -> tuple:
        return (9,)

    @classmethod
    def get_action_space(cls) -> int:
        return 9

    def render(self):
        if self.human_render:
            print()
            print("Enemy is", "X" if self.enemy_id == 1 else "O")
            for i in range(3):
                for j in range(3):
                    if self.board[i, j] == 0:
                        print("-", end=" ")
                    elif self.board[i, j] == 1:
                        print("X", end=" ")
                    elif self.board[i, j] == 2:
                        print("O", end=" ")
                    else:
                        print("\nInvalid board value")
                        print(self.board)
                        raise ValueError("Invalid board value")
                print()

    def get_obs(self) -> np.ndarray:
        return self.board.flatten()

    def check_win(self, player: int):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def check_tie(self):
        if np.all(self.board != 0) and not self.check_win(self.agent_id) and not self.check_win(self.enemy_agent):
            return True
        return False

    def get_action_mask(self):
        # A mask of 1s and 0s, where 1s indicate valid actions and 0s indicate invalid actions
        # Same shape as the action space
        # The board becomes a boolean array with True where the 0s are, and False where the 1s and 2s are. Then cast it to an int array with 1s and 0s
        # The result is an array with 1s where the board is empty, and 0s where the board is not empty
        action_mask = np.logical_not(self.board.flatten()).astype(int)
        return action_mask

    def render_message(self, message: str):
        if self.human_render:
            print(message)

    def human_render_sleep(self):
        if self.human_render:
            sleep(self.human_render_delay)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, np.ndarray | None]:
        self.human_render_sleep()
        self.render_message(f"Agent action: {action}")
        # The agent's turn
        x, y = action // 3, action % 3
        if self.board[x, y] != 0:  # Invalid action
            reward = -1
            done = True
        else:  # Valid action
            self.board[x, y] = self.agent_id
            self.render()
            if self.check_win(self.agent_id):  # Agent wins
                reward = 1
                done = True
                self.render_message("Agent wins")
                self.human_render_sleep()
            elif self.check_tie():  # Agent ties
                reward = 0.5
                done = True
                self.render_message("Tie")
            else:  # Enemy's turn
                enemy_action = self.enemy_agent.get_action(
                    self.get_obs(), self.get_action_mask(), training=False)
                self.human_render_sleep()
                self.render_message(f"Enemy action: {enemy_action}")
                x, y = enemy_action // 3, enemy_action % 3
                self.board[x, y] = self.enemy_id
                self.render()
                if self.check_win(self.enemy_id):  # Enemy wins
                    reward = -1
                    done = True
                    self.render_message("Enemy wins")
                    self.human_render_sleep()
                elif self.check_tie():  # Enemy ties
                    reward = 0.5
                    done = True
                    self.render_message("Tie")
                    self.human_render_sleep()
                else:  # The game continues
                    reward = 0
                    done = False
        return self.get_obs(), reward, done, self.get_action_mask()

    def reset(self) -> Tuple[np.ndarray, np.ndarray | None]:
        self.games_played += 1
        self.board = np.zeros((3, 3), dtype=int)
        # Switch who goes first
        self.agent_id = (self.agent_id % 2) + 1
        self.enemy_id = (self.enemy_id % 2) + 1
        # Swap the ids
        self.render()

        # Enemy
        if self.games_played % self.reload_enemy_interval == 0:
            self.enemy_agent.load()

        if self.enemy_id == 1:
            enemy_action = self.enemy_agent.get_action(
                self.get_obs(), self.get_action_mask(), training=False)
            self.render_message(f"Enemy action: {enemy_action}")
            x, y = enemy_action // 3, enemy_action % 3
            self.board[x, y] = self.enemy_id
            self.render()
        return self.get_obs(), self.get_action_mask()

    def seed(self, seed):
        pass
