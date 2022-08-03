# Made with Github Copilot
# The game Snake is a two-dimensional game where the goal is to eat as many apples as possible.
from typing import Tuple

import numpy

from envs.env import Env

# Constants
# Game object values
snake_head_value = 1
snake_body_value = 2
apple_value = 3
edge_value = 4


class Snake(Env):

    def __init__(self, grid_rows: int, grid_cols: int):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # Variables that are initialized in reset()
        self.grid = None
        self.snake_head_loc = None
        self.snake_body_locs = None
        self.apple_loc = None

    def step(self, action: int) -> Tuple[numpy.ndarray, float, bool]:
        pass

    def reset(self) -> numpy.ndarray:
        """
        Reset the environment and return an initial observation
        :return: The initial observation
        """
        # Initialize the grid
        self.grid = numpy.zeros((self.grid_rows, self.grid_cols))
        # Initialize the snake
        # The snake is initially one block long
        self.snake_head_loc = numpy.array([int(self.grid_rows / 2), int(self.grid_cols / 2)])
        self.snake_body_locs = numpy.array([])
        # Initialize the apple
        self.apple_loc = self.__spawn_apple()
        # Update the grid to contain the snake and apple
        self.grid[self.snake_head_loc[0], self.snake_head_loc[1]] = snake_head_value
        self.grid[self.apple_loc[0], self.apple_loc[1]] = apple_value
        # Return the initial observation
        return self.grid

    def get_observation_space(self) -> tuple:
        # Return the observation space
        # The observation space is a grid of the same size as the grid
        # The observation space contains the following values:
        # 0: empty
        # 1: snake head
        # 2: snake body
        # 3: apple
        # 4: edge
        return 1, 1, self.grid_rows, self.grid_cols

    def get_action_space(self) -> int:
        return 4  # Up, down, left, right

    def render(self):
        pass

    def seed(self, seed):
        pass

    def __spawn_apple(self):
        """
        Spawns an apple in a random location on the grid
        :return: The location of the apple
        """
        # Keep looping until we find a valid spawn location
        while True:
            # Generate a random location
            apple_loc = numpy.array([numpy.random.randint(0, self.grid_rows), numpy.random.randint(0, self.grid_cols)])
            # Check if this location is valid
            if not self.__is_occupied(apple_loc):
                return apple_loc

    def __is_occupied(self, loc: numpy.ndarray) -> bool:
        """
        Checks if the given location is occupied by a snake body or apple
        :param loc: The location to check
        :return: True if the location is occupied, False otherwise
        """
        # Check if the location is occupied by the snake head
        if numpy.array_equal(loc, self.snake_head_loc):
            return True
        # Check if the location is occupied by the snake body
        for snake_body_loc in self.snake_body_locs:
            if numpy.array_equal(snake_body_loc, loc):
                return True
        # Check if the location is occupied by the apple
        if numpy.array_equal(self.apple_loc, loc):
            return True
        # The location is not occupied
        return False
