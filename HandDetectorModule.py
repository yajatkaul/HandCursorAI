import cv2
import mediapipe as mp
import time
import math
import autopy as ap
import numpy as np
import pyautogui as pg

frameR = 100 # Frame Reduction
wCam, hCam = 640, 480

#names are sens like handLms not HandLMS
cap = cv2.VideoCapture(1) #webcam select
cap.set(3, wCam)
cap.set(4, hCam)
wScr, hScr = ap.screen.size()

mpHands = mp.solutions.hands             #mp.solutions.hands - provides hand detection
hands = mpHands.Hands(False)                   # then has to call hand by using mpHands.Hand()
mpDraw = mp.solutions.drawing_utils       #mp.solutions.drawing_utils - provides drawing tools

pTime = 0
cTime = 0

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #line-16 Convert color mode
    results = hands.process(imgRGB)    
    fingerlst=[0,0,0,0,0]
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),(255, 0, 255), 2)
    if(results.multi_hand_landmarks): #when hand is detected
        for handLms in results.multi_hand_landmarks:   #use for loop to get each hand  here handLms - hand landmarks


            lmlist = []
            for id,lm in enumerate(handLms.landmark):

                h, w, c = img.shape   
                cx , cy = int(lm.x*w),int(lm.y*h)
                lmlist.append([id,cx,cy])
            if(len(lmlist)!=0):
                if(lmlist[8][2] < lmlist[6][2]):  #finger index
                    fingerlst[1] = 1
                else:
                    fingerlst[1] = 0
                if(lmlist[12][2] < lmlist[10][2]):
                    fingerlst[2] = 1
                else:
                    fingerlst[2] = 0
                if(lmlist[16][2] < lmlist[14][2]):
                    fingerlst[3] = 1
                else:
                    fingerlst[3] = 0
                if(lmlist[20][2] < lmlist[18][2]):
                    fingerlst[4] = 1
                else:
                    fingerlst[4] = 0
                if(lmlist[4][1] > lmlist[2][1]):
                    fingerlst[0] = 1
                else:
                    fingerlst[0] = 0

            if(fingerlst[1] == 1 and fingerlst[2] == 1 and fingerlst[3] == 1 and fingerlst[4] == 1 and fingerlst[0] == 1):
                ap.mouse.toggle(ap.mouse.Button.LEFT, True)    
                x1,y1 = lmlist[8][1],lmlist[8][2]
                x3 = np.interp(x1, (0, wCam), (0, wScr))
                y3 = np.interp(y1, (0, hCam), (0, hScr))
                print(wScr - x3,y3)
                ap.mouse.move(wScr - x3,y3)    
            elif(fingerlst[1] == 1 and fingerlst[2] == 1 and fingerlst[3] == 1 and fingerlst[4] == 1):
                pg.scroll(-100)
            elif(fingerlst[1] == 1 and fingerlst[2] == 1 and fingerlst[3] == 1):
                pg.scroll(100)
            elif(fingerlst[1] == 1 and fingerlst[2] == 1):
                x1,y1 = lmlist[8][1],lmlist[8][2]
                x2,y2 = lmlist[12][1],lmlist[12][2]
                distbtwn = math.hypot(x2-x1,y2-y1)
                if(distbtwn < 27):
                    ap.mouse.click()
            elif(fingerlst[1] == 1):
                x1,y1 = lmlist[8][1],lmlist[8][2]
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                print(wScr - x3,y3)
                ap.mouse.move(wScr - x3,y3)
            elif(fingerlst == [0,0,0,0,0]):
                ap.mouse.toggle(ap.mouse.Button.LEFT, False)  


            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   #draw_landmarks for drawuing the dots takes "img" - video , "handLms" - hand, "HAND_CONNECTIONS" for drawing lines on hand
    
    cTime = time.time()      #
    fps = 1/(cTime - pTime)  # fps calc
    pTime = cTime            #

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255), 3)  #display text on top left using putText - parameters -> videoshown , data(in string),position,font,font size,color,thickness   --------------- data3cf
    
    cv2.imshow("Image",img) #cam to display
    cv2.waitKey(1)





