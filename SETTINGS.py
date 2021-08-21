import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import mediapipe as md
import math
import pyautogui as pag

wCam, hCam = 1280, 720
wScreen, hScreen = pag.size()
cTime = 0
pTime = 0
pag.FAILSAFE = False
stp = 0
start = False
FrameRed = 200
smoothening = 6
plocX, plocY = 0, 0
clocX, clocY = 0, 0
a = 0
b = 0
