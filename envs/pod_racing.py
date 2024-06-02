import pygame
import numpy as np
import math
import sys

from envs.resources import *

class RaceTrackEnv():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}

    def __init__(self, render_mode=None, num_players = 1, num_bots = 1, num_laps = None, render_fps = 6):

        assert num_players + num_bots > 0, "Not enough pods to play the game"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.num_players = num_players
        self.num_bots = num_bots
        self.num_laps = num_laps
        self.metadata["render_fps"] = render_fps

        
    # The step function takes a 2D array of actions (x, y, thrust), with number of columns equal to the number of players. 
    # It will then iterate through the list of players and bots:
    #   -for each player it will assign the same-indexed action from the actions array as its (x, y, thrust),
    #   -for each bot it will assign the postion of the next checkpoint as its (x, y) and 100 as its thrust.
    #
    # For all pods it will use the assigned action to calculate the new position. 
    #
    # the function will return:
    #
    #   -observations, which is a 2d array:
    #       first row contaling a list of tuples indicating each player location,
    #       second row containg a list of tuples indicating the target checkpoint for each player.
    #       each with number of columns equal to the number of players.
    #       
    #       for example, starting 3 players would return observations:
    #       [[(12000, 1990), (12000, 1990), (12000, 1990)], [(10680, 4990), (10680, 4990), (10680, 4990)]]
    #
    #   -rewards, which is an array containing the rewards for each player.
    #
    #   -terminated, which indicates if the game ended.
    #
    #   -info, wich is an array of tuples containig the location of all players and all bots.

    def step(self, actions):
        assert len(actions) == self.num_players, "Number of actions doesnt match number of players"
        
        rewards = []
        lines = []

        for pod in self.pods:
            # Players have a numberd id, bot have "bot" as their id
            if isinstance(pod.id, int):
                x, y, thrust = actions[pod.id]
                assert 0 <= x <= WIDTH and 0 <= y <= HEIGHT and 0 <= thrust <= 101, "Action out of range"
                
                last_distance = checkDistance(*pod.get_pos(), *self.checkpoints[pod.target])
                lines.append([x,y])
            else:
                thrust = 100
                x, y = self.checkpoints[pod.target]
            
            pod.update(x, y, thrust)

            # Checkpoint counter for each pod
            if checkDistance(*pod.get_pos(), *self.checkpoints[pod.target]) < 800:
                pod.target = (pod.target + 1) % len(self.checkpoints)
                pod.checked += 1

            # The number of checkpoints a pod corssed, divided by 4 is the number of laps.
            # If a pod completes the total number of laps, the game is over. 
            if self.num_laps != None and (pod.checked / 4) >= self.num_laps:
                self.terminated = True

            # it calcualtes a reward based on the distance to the next checkpoint:
            #   if the distance didnt shrink, the rewards is -1
            #   if the distance did get smaller (i.e. the pod got closer to the target), the pod receives a reward
            if isinstance(pod.id, int):
                reward = -1

                if last_distance > checkDistance(*pod.get_pos(), *self.checkpoints[pod.target]):
                    reward = (last_distance - checkDistance(*pod.get_pos(), *self.checkpoints[pod.target]))/(185)

                if pod.checked == 1:
                    reward = 100
                    pod.checked = 0

                rewards.append(reward)

        # Render the new frame
        if self.render_mode == "human":
            self._render_frame(lines)

        observations = get_obs(self.pods, self.num_players, self.checkpoints)
        info = get_info(self.pods)

        return observations, rewards, self.terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    # The render frame function works with pygame to render the frames of the environement. 
    # Each checkpoint is drawn as a circle, and each pod is drawn as triangle.    
    # Additionally, 2 lines are drawn per player pod:
    #   -One short green line aiding the visualization of direction of travel.
    #   -One white line showing the selected point to travel to.
    def _render_frame(self, *args):
        lines = args[0]

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WIDTH/10,HEIGHT/10))
            pygame.display.set_caption(TITLE)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((WIDTH/10,HEIGHT/10))
        canvas.fill((25, 25, 25))

        font = pygame.font.Font('freesansbold.ttf', 32)
        
        numbers = []
        rects = []

        # Load checkpoints
        for checkpoint in self.checkpoints:
            pygame.draw.circle(canvas, BLUE, (checkpoint[0]/10, checkpoint[1]/10), RADIUS)
            numbers.append(font.render(str(self.checkpoints.index(checkpoint)), True, WHITE))
            rects.append(numbers[-1].get_rect())
            rects[-1].center = (checkpoint[0]/10, checkpoint[1]/10)

        # Load pods
        for pod in self.pods:
            modifiers = get_triangle(pod.angle)
            result = tuple((mod_x + pod.x/10, mod_y + pod.y/10) for mod_x, mod_y in modifiers)
            pygame.draw.polygon(canvas, pod.color, result)

       # Load lines
        for i in range(self.num_players):
            pod = self.pods[i]
            pos = ((pod.x + math.cos(pod.angle) * 1000) / 10, (pod.y + math.sin(pod.angle) * 1000)/10)

            pygame.draw.line(canvas,WHITE, (pod.x /10, pod.y /10), ((lines[i][0])/10, (lines[i][1])/10))
            pygame.draw.line(canvas,GREEN, (pod.x /10, pod.y /10), pos)

        # Draw all loaded elements
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for _ in range(len(numbers)):
                self.window.blit(numbers[_], rects[_])
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                
                # checking if key "q" was pressed
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()


    # The reset function will create:
    #
    #   an array for the checkpoints (wich is fixed at the moment)
    #   a list of player pods, of length equal to num_players
    #   a list of bot pods, of length equal to num_bots
    #
    # and it will return:
    #
    #   observations, which is a 2d array:
    #       first column contaling a list of tuples indicating each player location
    #       second column containg a list of tuples indicating the target checkpoint for each player
    #       
    #       for example, starting 3 players would return observations:
    #       [[(12000, 1990), (12000, 1990), (12000, 1990)], [(10680, 4990), (10680, 4990), (10680, 4990)]]
    #
    #   info, wich is an array of tuples containig the location of all players and all bots
    def reset(self, **kwargs):
        
        self.terminated = False

        self.pods = []
        self.checkpoints = [(12000, 1990), (10680, 4990), (14020, 3010), (3990, 7780)]
        self.pods += [Pod(i, *self.checkpoints[0], *self.checkpoints[1], 1, np.random.randint(256, size=3)) for i in range(self.num_players)]
        self.pods += [Pod("bot", *self.checkpoints[0], *self.checkpoints[1], 1, np.random.randint(256, size=3)) for i in range(self.num_bots)]

        observations = get_obs(self.pods, self.num_players, self.checkpoints)
        info = get_info(self.pods)

        return observations, info
    
    def close(self):
        print("closed")
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
