# Gym environment for the game of Flappy Bird
import random
from collections import deque
from typing import Tuple

import numpy as np
import pygame


class Bird:
    """
    The bird class
    The bird has a position and a velocity.
    The bird can jump and fall.
    The bird can be controlled by the agent.
    The bird does not move horizontally.
    """

    def __init__(self):
        #
        # The bird state
        #
        # The bird's position
        self.x = 100
        self.y = 0
        # The bird's size
        self.width = 40
        self.height = 40
        # The bird's velocity
        self.vy = 0
        # The bird's acceleration
        self.ay = 0
        self.gravity = 0.015
        self.max_vy = 10

        #
        # Bird represented as a circle
        #
        self.image = pygame.Surface((self.width, self.height))
        # Transparent image
        self.image.set_alpha(0)
        self.bird_color = (255, 255, 255)
        # The circle is drawn when rendering the bird

    def jump(self):
        """
        The bird jumps
        """
        self.vy = -6
        self.ay = 0

    def fall(self):
        """
        The bird falls
        """
        self.vy += self.ay
        self.y += self.vy
        self.ay += self.gravity
        # Check if the bird is falling too fast
        if self.vy > self.max_vy:
            self.vy = self.max_vy

    def update(self, action: int):
        """
        Update the bird's position and velocity
        """
        if action == 1:
            self.jump()
        else:
            self.fall()

    def render(self, screen: pygame.Surface):
        """
        Render the bird on the screen
        """
        # Draw the bird on the screen
        screen.blit(self.image, (self.x, self.y))
        # Draw a circle to represent the bird
        pygame.draw.circle(screen, self.bird_color, (self.x + self.width // 2, self.y + self.height // 2),
                           self.height // 2)

    def get_state(self) -> Tuple[float, float]:
        """
        Get the bird's state
        """
        return self.x, self.y

    def reset(self):
        self.vy = 0
        self.ay = 0


class Pipe:
    """
    The pipe class
    """

    def __init__(self):
        #
        # The pipe state, initialized by the game
        #
        # The pipe's position
        self.x = None
        self.y = None
        # The pipe's size
        self.width = None
        self.height = None
        # The pipe's velocity
        self.vx = None
        # The pipe's acceleration
        self.ax = None
        self.reset()
        self.pipe_color = (0, 100, 0)

    def update(self):
        """
        Update the pipe's position and velocity
        """
        self.x += self.vx
        self.vx += self.ax

    def render(self, screen: pygame.Surface):
        """
        Render the pipe on the screen
        """
        pygame.draw.rect(screen, self.pipe_color, (int(self.x), int(self.y), self.width, self.height))

    def get_state(self) -> Tuple[float, float]:
        """
        Get the pipe's state
        """
        return self.x, self.y

    def set_state(self, x: float, y: float, width: float, height: float):
        """
        Set the pipe's state
        The state is a tuple of the pipe's x position, y position, width, height
        """
        self.x, self.y, self.width, self.height = x, y, width, height

    def reset(self):
        self.vx = -5
        self.ax = 0


class FlappyBird:
    """
    The game class
    The game is responsible for the rendering.
    The game is responsible for the state.
    The game is responsible for the action.
    The game is responsible for the reward.
    The game is responsible for the done.

    There are two types of pipes:
    1. The top pipe
    2. The bottom pipe

    The top pipe is always moving to the left.
    The bottom pipe is always moving to the left.

    The bird is stationary.

    Pipes spawn at a random heights.
    A pair of top and bottom pipes spawn off the left side of the screen.
    Once the last spawned pair of pipes move a certain distance, a new pair of pipes is spawned off the screen.
    Once the pipes spawn, they move to the left.
    Once the pipes move off the screen,
    they are moved back to the right of the screen and are given a new random height.

    The game is done when the bird hits the ground or the ceiling.
    The game is done when the bird hits the top pipe.
    The game is done when the bird hits the bottom pipe.
    The game is done when the bird hits the pipes.
    """

    def __init__(self):
        # The pygame window
        self.screen = None
        # The window size
        self.window_width = 800
        self.window_height = 800
        # The bird
        self.bird = Bird()
        # Move the bird horizontally near to the center of the first third of the screen
        self.bird.x = self.window_width / 3
        # The score
        self.score = 0
        # The score font color
        self.score_color = (0, 0, 0)
        # The index of the pipe pair that the bird needs to pass to get a reward and increase the score
        self.current_pair_index = 0
        # Pipes
        # The pipe pairs are a deque of pipes,
        # so that the front of the deque is the pipe that is closest to the left side of the screen,
        # and the back of the deque is the pipe that is farthest from the left side of the screen.
        # Once the left most pipe pair leaves the screen,
        # it is removed from the deque and added to the back of the deque with a new random position
        self.pipe_pairs = deque()
        self.pipe_distance = 300
        self.pipe_gap = 150
        self.pipe_width = 100
        self.pipe_height = 1000
        # This offset is used to make the place the gap between the pipes
        # not too close to the top and bottom of the screen
        self.pipe_gap_offset = 50
        # sky blue background
        self.background_color = (135, 206, 250)
        # The font used to display the score
        # Must initialize the pygame font module
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 30)
        # Setting used to slow down the game for human play
        self.human_mode = False
        self.reset()

    def reset(self):
        # Reset the game
        self.bird.reset()
        # Move the bird vertically near to the center of the screen
        self.bird.y = self.window_height / 2
        self.pipe_pairs = deque()
        self.score = 0
        # The index of the pipe pair that the bird needs to pass to get a reward and increase the score
        self.current_pair_index = 0
        # Pipes
        self.create_pipe_pair()

        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        # Update the game
        self.update(action)
        # Delay if in human mode
        if self.human_mode:
            pygame.time.wait(15)
        # Get the state
        state = self.get_state()
        # Get the reward
        reward = self.get_reward()
        # Check if the game is done
        done = self.is_done()
        return state, reward, done

    def get_observation_space(self) -> int:
        pass

    def get_action_space(self) -> int:
        pass

    def render(self):
        # Render the game
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        # Draw the background
        self.screen.fill(self.background_color)
        self.bird.render(self.screen)
        for top, bottom in self.pipe_pairs:
            top.render(self.screen)
            bottom.render(self.screen)
        # Draw the score
        score_text = self.font.render(f"Score: {self.score}", True, self.score_color)
        self.screen.blit(score_text, (10, 10))
        # Update the screen
        pygame.display.flip()


    def get_state(self) -> np.ndarray:
        # Get the state
        state = np.zeros((1, 1))
        state[0, 0] = self.bird.y
        return state

    def update(self, action: int):
        self.bird.update(action)
        for top, bottom in self.pipe_pairs:
            top.update()
            bottom.update()

        # Check if the bird passes the current pipe pair
        current_top, current_bottom = self.pipe_pairs[self.current_pair_index]
        if current_top.x + current_top.width < self.bird.x:
            # The bird passed the current pipe pair
            # Increase the score
            self.score += 1
            # Move to the next pipe pair
            self.current_pair_index += 1
        # Check if the oldest pair of pipes has left the screen,
        # and if so, move it to the back of the deque and rearrange its position
        oldest_top, oldest_bottom = self.pipe_pairs[0]
        if oldest_top.x + oldest_top.width < 0:
            pair = self.pipe_pairs.popleft()
            top, bottom = pair
            self.arrange_pipe_pair(top, bottom)
            self.pipe_pairs.append(pair)
            # Adjust the current pair index
            self.current_pair_index -= 1

        # Check if the last pair of pipes has moved a certain distance, and if so, create a new pair of pipes
        last_top, last_bottom = self.pipe_pairs[-1]
        if last_top.x + last_top.width < self.window_width - self.pipe_distance:
            self.create_pipe_pair()

    def get_reward(self):
        # Get the reward

        # If the game is done, return a reward of -1, otherwise return 1
        if self.is_done():
            return -1
        else:
            return 1

    def is_done(self) -> bool:
        # Check if the game is done
        # Check if the bird is out of the screen
        if self.bird.y < 0 or self.bird.y > self.window_height:
            return True
        # Check if the bird hits the ground
        if self.bird.y + self.bird.height > self.window_height:
            return True
        # Check if the bird hits the ceiling
        if self.bird.y < 0:
            return True
        # Check if the bird hits a pipe
        for top, bottom in self.pipe_pairs:
            # Check if the bird hits the top pipe
            # using the bird_intersects_pipe function
            if self.bird_intersects_pipe(top):
                return True
            # Check if the bird hits the bottom pipe
            if self.bird_intersects_pipe(bottom):
                return True
        return False

    def bird_intersects_pipe(self, pipe: Pipe) -> bool:
        # Check if the bird hits a pipe using
        # the bird's x, y, width, and height
        # And the pipe's x, y, width, and height
        return self.bird.x + self.bird.width > pipe.x and self.bird.x < pipe.x + pipe.width and self.bird.y + self.bird.height > pipe.y and self.bird.y < pipe.y + pipe.height

    def arrange_pipe_pair(self, top_pipe: Pipe, bottom_pipe: Pipe):
        # Arrange the pipes in a pair
        # If the deque is empty, position the pipes just off the right side of the screen
        # If the deque is not empty, position the pipes the pipe_distance away from the rightmost pair of pipes
        if len(self.pipe_pairs) == 0:
            top_pipe.x = self.window_width
            bottom_pipe.x = self.window_width
        else:
            right_most_top, right_most_bottom = self.pipe_pairs[-1]
            top_pipe.x = right_most_top.x + right_most_top.width + self.pipe_distance
            bottom_pipe.x = right_most_bottom.x + right_most_top.width + self.pipe_distance
        # Position the top pipe at a random height between the top of the screen + the gap offset + the gap size
        # and the bottom of the screen - the gap offset - the gap size
        top_pipe.y = -self.pipe_height + random.randint(self.pipe_gap_offset + self.pipe_gap, self.window_height - self.pipe_gap_offset - self.pipe_gap)
        # Position the bottom pipe at the same height as the top pipe
        bottom_pipe.y = top_pipe.y + self.pipe_gap + self.pipe_height

    def create_pipe_pair(self):
        bottom_pipe = Pipe()
        top_pipe = Pipe()
        # Set the width and height of the pipes
        bottom_pipe.width = self.pipe_width
        bottom_pipe.height = self.pipe_height
        top_pipe.width = self.pipe_width
        top_pipe.height = self.pipe_height
        # Arrange the pipes in a pair
        self.arrange_pipe_pair(top_pipe, bottom_pipe)
        # Add the pair to the deque
        self.pipe_pairs.append((top_pipe, bottom_pipe))

    def set_human_mode(self, human_mode: bool):
        # Make the game loop delay for human players if true
        self.human_mode = human_mode
