import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, QUIT, K_q
import sys
import cv2

W = 1920//2
H = 1080//2

pygame.init()
screen = pygame.display.set_mode((W, H))


def multithreading(img):
    img = cv2.resize(img, (W, H))

    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1)).convert()
    print(surf)

    screen.blit(surf, (0, 0))
    pygame.display.update()

    print(img.shape)


if __name__ == '__main__':
    cap = cv2.VideoCapture('test.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            multithreading(frame)
        else:
            break