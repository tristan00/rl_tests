
import os, sys
import pygame
from pygame.locals import *


class Snake():
    pass


class Board():
    def __init__(self, board_size = 256, square_size = 4):
        self.x = board_size//square_size
        self.y = board_size // square_size

        