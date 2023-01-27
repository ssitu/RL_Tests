# Environment for the game of Flappy Bird
import random
import time
from abc import ABC
from collections import deque
from typing import Tuple

import numpy as np
import pygame

from envs.env import Env

TRUNCATE_MAX_SCORE = 5
BIRD_MAX_VELOCITY = 10
BIRD_COLOR = (172, 57, 49)
PIPE_COLOR = (54, 65, 82)
BACKGROUND_COLOR = (40, 44, 52)
SCORE_COLOR = (150, 150, 150)
FRAME_DELAY = 15


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

        #
        # Bird represented as a circle
        #
        self.image = pygame.Surface((self.width, self.height))
        # Transparent image
        self.image.set_alpha(0)
        # The circle is drawn when rendering the bird

    def jump(self):
        """
        The bird jumps
        """
        self.vy = -6
        self.ay = 0

    def update(self, action: int):
        """
        Update the bird's position and velocity
        """
        if action == 1:
            self.jump()

        # Update the bird's position
        self.y += self.vy
        self.vy += self.ay
        self.ay += self.gravity
        # Check if the bird is falling too fast
        if self.vy > BIRD_MAX_VELOCITY:
            self.vy = BIRD_MAX_VELOCITY

    def render(self, screen: pygame.Surface):
        """
        Render the bird on the screen
        """
        # Draw the bird on the screen
        screen.blit(self.image, (self.x, self.y))
        # Draw a circle to represent the bird
        pygame.draw.circle(screen, BIRD_COLOR, (self.x + self.width // 2, self.y + self.height // 2),
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
        pygame.draw.rect(screen, PIPE_COLOR, (int(self.x), int(self.y), self.width, self.height))

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
        self.vx = -4
        self.ax = 0


class FlappyBird(Env, ABC):
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

    def __init__(self, human_render: bool = False, truncate=True, fastest_speed=False, human_player=False):
        super().__init__(human_render)
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
        # The index of the pipe pair that the bird needs to pass to get a reward and increase the score
        self.current_pair_index = 0
        # Pipes
        # The pipe pairs are a deque of pipes,
        # so that the front of the deque is the pipe that is closest to the left side of the screen,
        # and the back of the deque is the pipe that is farthest from the left side of the screen.
        # Once the left most pipe pair leaves the screen,
        # it is removed from the deque and added to the back of the deque with a new random position
        self.pipe_pairs = deque()
        self.pipe_distance = 400
        self.pipe_gap = 200
        self.pipe_width = 100
        self.pipe_height = 1000
        # This offset is used to make the place the gap between the pipes
        # not too close to the top and bottom of the screen
        self.pipe_gap_offset = 100
        # The font used to display the score
        # Must initialize the pygame font module
        pygame.font.init()
        self.font = pygame.font.SysFont('Arial', 30)
        # Random number generator
        self.rng = random.Random()
        self.high_score = 0
        self.truncate = truncate
        self.last_time = None
        self.fastest_speed = fastest_speed
        # The human player
        self.human_player = human_player
        self.reset()

    def reset(self):
        # Reset the game
        self.bird.reset()
        # Move the bird vertically near to the center of the screen
        self.bird.y = self.window_height / 2
        self.pipe_pairs = deque()
        if self.score > self.high_score:
            self.high_score = self.score
            print("New high score: {}".format(self.high_score))
        self.score = 0
        # The index of the pipe pair that the bird needs to pass to get a reward and increase the score
        self.current_pair_index = 0
        # Pipes
        self._create_pipe_pair()
        self.last_time = time.time()

        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        #
        # Call to events to prevent the game from freezing
        #
        # Must initialize pygame video system
        pygame.display.init()
        # Call to events
        pygame.event.pump()

        if self.human_render:
            self.render()
            delta_time = time.time() - self.last_time
            if not self.fastest_speed:
                pygame.time.wait(FRAME_DELAY - int(delta_time * 1000))
            self.last_time = time.time()
        # Human player
        if self.human_player:
            # Check for keys pressed
            keys = pygame.key.get_pressed()
            # If the space bar is pressed, flap
            if keys[pygame.K_SPACE]:
                action = 1
            else:
                action = 0
        # Update the game
        self._update(action)
        # Get the state
        state = self._get_state()
        # Get the reward
        reward = self._get_reward()
        # Check if the game is done
        done = self._is_terminated() or self._is_truncated()
        return state, reward, done

    def get_observation_space(self) -> tuple:
        # This is the length of the state vector
        return len(self._get_state()),

    def get_action_space(self) -> int:
        # The action space is 0 for do nothing, 1 for flap
        return 2

    def render(self):
        # Render the game
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        # Draw the background
        self.screen.fill(BACKGROUND_COLOR)
        self.bird.render(self.screen)
        for top, bottom in self.pipe_pairs:
            top.render(self.screen)
            bottom.render(self.screen)
        # Draw the score
        score_text = self.font.render(f"Score: {self.score}", True, SCORE_COLOR)
        self.screen.blit(score_text, (10, 10))
        # Draw the high score
        high_score_text = self.font.render(f"High score: {self.high_score}", True, SCORE_COLOR)
        self.screen.blit(high_score_text, (10, 50))
        # Update the screen
        pygame.display.flip()

    def seed(self, seed):
        # Seed the random number generator
        self.rng.seed(seed)

    def _get_state(self) -> np.ndarray:
        # Get the state
        # Scale down the bird's y
        bird_y = self.bird.y / self.window_height
        # Scale down the bird's y velocity
        bird_vy = self.bird.vy / BIRD_MAX_VELOCITY

        # The next pipe pair that the bird will pass through
        _, bottom = self.pipe_pairs[self.current_pair_index]
        # Scale down the next pipes' x
        pipe_x = bottom.x / self.window_width
        # Scale down the next pipes' y
        pipe_y = bottom.y / self.window_height

        # The following pipe pair after the current one
        following = self.current_pair_index + 1
        following_pipe_x = 0
        following_pipe_y = 0
        if following < len(self.pipe_pairs):
            _, following_bottom = self.pipe_pairs[following]
            # Scale down the pipes' x
            following_pipe_x = following_bottom.x / self.window_width
            # Scale down the pipes' y
            following_pipe_y = following_bottom.y / self.window_height

        return np.array([bird_y, bird_vy, pipe_x, pipe_y, following_pipe_x, following_pipe_y])

    def _update(self, action: int):
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
            self._arrange_pipe_pair(top, bottom)
            self.pipe_pairs.append(pair)
            # Adjust the current pair index
            self.current_pair_index -= 1

        # Check if the last pair of pipes has moved a certain distance, and if so, create a new pair of pipes
        last_top, last_bottom = self.pipe_pairs[-1]
        if last_top.x + last_top.width < self.window_width - self.pipe_distance:
            self._create_pipe_pair()

    def _get_reward(self):
        # Get the reward

        # If the game is done, return a reward of -1, otherwise return 1
        if self._is_terminated():
            return -1
        else:
            return .1

    def _is_terminated(self) -> bool:
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
            if self._bird_intersects_pipe(top):
                return True
            # Check if the bird hits the bottom pipe
            if self._bird_intersects_pipe(bottom):
                return True
        return False

    def _is_truncated(self):
        if self.truncate and self.score >= TRUNCATE_MAX_SCORE:
            return True

    def _bird_intersects_pipe(self, pipe: Pipe) -> bool:
        # Check if the bird hits a pipe using
        # the bird's x, y, width, and height
        # And the pipe's x, y, width, and height
        return self.bird.x + self.bird.width > pipe.x \
               and self.bird.x < pipe.x + pipe.width \
               and self.bird.y + self.bird.height > pipe.y \
               and self.bird.y < pipe.y + pipe.height

    def _arrange_pipe_pair(self, top_pipe: Pipe, bottom_pipe: Pipe):
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
        top_pipe.y = -self.pipe_height + self.rng.randint(self.pipe_gap_offset,
                                                          self.window_height - self.pipe_gap_offset - self.pipe_gap)
        # Position the bottom pipe at the same height as the top pipe
        bottom_pipe.y = top_pipe.y + self.pipe_gap + self.pipe_height

    def _create_pipe_pair(self):
        bottom_pipe = Pipe()
        top_pipe = Pipe()
        # Set the width and height of the pipes
        bottom_pipe.width = self.pipe_width
        bottom_pipe.height = self.pipe_height
        top_pipe.width = self.pipe_width
        top_pipe.height = self.pipe_height
        # Arrange the pipes in a pair
        self._arrange_pipe_pair(top_pipe, bottom_pipe)
        # Add the pair to the deque
        self.pipe_pairs.append((top_pipe, bottom_pipe))
