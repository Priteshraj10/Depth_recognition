import pygame
from pygame.locals import DOUBLEBUF
from multiprocessing import process
import pangolin
import OpenGL.GL as gl
import numpy as np


class Display2D(object):
    def __init__(self, W, H):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img):
        for event in pygame.event.get():
            pass
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0, 1)[:, :, [0, 1, 2]])
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()


class Display3D(object):
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process()
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGLRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0, 0, 0,
                                     0, -1, 0))

        self.handler = pangolin.Handler3D(self.scam)
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, w/h)
        self.dcam.Resize(pangolin.Viewport(0, 0, w*2, h*2))
        self.dcam.Activate()

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        gl = glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state is not None:
            if self.state[0].shape[0] >=2:
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawCameras(self.state[0][:-1])

            if self.state is not None:



