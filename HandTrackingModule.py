import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # Hand tracking module
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    for id, lm in enumerate(handLms.landmark):
        # print(id,lm)
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        print(id, cx, cy)
        if id == 0:
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    def findPosition(self,img,handNo=0,draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                    #print(id,lm)
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    #print(id,cx,cy)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return lmList
def main():
    pTime = 0
    cTime = 0

    # VideoCapture(0) is the default camera
    cap = cv2.VideoCapture(1)

    while True:
        success, img = cap.read()

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # Display FPS
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()