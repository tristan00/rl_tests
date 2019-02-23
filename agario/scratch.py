import os, sys
import pygame
import random
screen_size = 512

class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, screen, color):
        super(Agent, self).__init__()
        self.x = x
        self.y = y
        self.screen = screen


    def redraw(self):
        pygame.draw.circle(self.screen, (128, 128, 128), (self.x, self.y), 5)


    def random_move(self):
        self.x = self.x + random.choice([-1, 1])
        self.y = self.y + random.choice([-1, 1])


class Game():

    def __init__(self, screen, agent_count = 10, food_count = 100, max_rounds = 1000):
        self.agents = [Agent(x = random.uniform(16, screen_size - 16), y = random.uniform(16, screen_size - 16), color = (int(255 - (i * (agent_count / 255))), int(255 - (i * (agent_count / 1000))), int((i * (agent_count / 255)))), screen = screen) for i in range(agent_count)]





def run_game():
    pygame.init()
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption('test')
    pygame.mouse.set_visible(0)
    screen.fill((0, 0, 0))

    a = Agent(128, 128, screen)

    for i in range(10000):
        a.random_move()
        a.redraw()
        pygame.display.flip()
        screen.fill((0, 0, 0))
        print('here')



run_game()