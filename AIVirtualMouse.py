import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import time
import mediapipe as md
import pyautogui as pag
from pynput.mouse import Button, Controller
import math
import SETTINGS as st

######################
wCam, hCam = st.wCam, st.hCam
wScreen, hScreen = st.wScreen, st.hScreen
cTime = st.cTime
pTime = st.pTime
FrameRed = st.FrameRed
smoothening = st.smoothening
plocX, plocY = st.plocX, st.plocY
clocX, clocY = st.clocX, st.clocY
pag.FAILSAFE = False
a = st.a
b = st.b
######################

cam = cv.VideoCapture(0)

cam.set(3, wCam)
cam.set(4, hCam)

detector = htm.hand_detector(min_det_con=0.7, max_hands_num=1)

mouse = Controller()

while True:
    if a > 0:
        a -= 1
    if b > 0:
        b -= 1
    success, frame = cam.read()
    frame = cv.flip(frame, 1)
    frame = detector.find_hands(frame)
    lmList, bbox = detector.find_pos(frame, draw=True)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, " ", y1, " ", x2, " ", y2)

        fingers = detector.fingers_up()
        # print(fingers)

        if fingers[1] == 1 and fingers[2] == 0 and fingers[4] == 0:
            cv.rectangle(frame, (FrameRed, FrameRed), (wCam - FrameRed, hCam - FrameRed), (255, 0, 255), 2)
            x3 = np.interp(x1, (FrameRed, wCam - FrameRed), (0, wScreen))
            y3 = np.interp(y1, (FrameRed, hCam - FrameRed), (0, hScreen))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            mouse.position = (clocX, clocY)
            cv.circle(frame, (x1, y1), 15, (255, 0, 255), cv.FILLED)
            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1 and fingers[4] == 0:
            length, frame, line_info = detector.find_dist(8, 12, frame)
            # print(length)
            if length < 35:
                cv.circle(frame, (line_info[4], line_info[5]), 15, (0, 255, 0), cv.FILLED)
                if a == 0:
                    pag.click()
                    a += 7

        if fingers[1] == 0 and fingers[2] == 1 and fingers[4] == 1:
            length3, frame, line_info3 = detector.find_dist(12, 20, frame)
            # print(length)
            if length3 < 100:
                cv.circle(frame, (line_info3[4], line_info3[5]), 15, (0, 255, 0), cv.FILLED)
                if b == 0:
                    pag.click(button='right')
                    b += 7

        if fingers[1] == 1 and fingers[4] == 1 and fingers[2] == 0:
            length2, frame, line_info2 = detector.find_dist(8, 20, frame)
            print(length2)
            if length2 < 160:
                pag.scroll(40)
            else:
                pag.scroll(-40)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, f'FPS: {int(fps)}', (10, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv.imshow("MouseControl", frame)

    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
