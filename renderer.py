import sys
import time

sys.path.append("lib/macosx")
sys.path.append("lib/linux")

from helpers import add_ones
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

np.set_printoptions(suppress=True)

class Renderer(object):
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.vertices = (
            (1, -1, -1), (1, 1, -1),
        )