import cv2 as cv
import mediapipe as mp
import time

cam = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cTime = 0
pTime = 0

while True:
    success, frame = cam.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                # if id == 4:
                #    cv.circle(frame, (cx, cy), 15, (255, 0, 255), cv.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10, 40), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    cv.imshow("HandTrack", frame)
    if cv.waitKey(1) == 27:
        break

cam.release()
cv.destroyAllWindows()
