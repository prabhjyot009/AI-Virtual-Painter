import cv2
import time
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2,modelC=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelC, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLand in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLand, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.PosList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.PosList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return self.PosList

    def fingerUp(self):
        fingers = []
        if len(self.PosList) >= self.tipIds[0] and len(self.PosList) >= self.tipIds[0] - 1:
            if self.PosList[self.tipIds[0]][1] < self.PosList[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        for id in range(1, 5):
            if len(self.PosList) >= self.tipIds[id] and len(self.PosList) >= self.tipIds[id] - 2:
                if self.PosList[self.tipIds[id]][2] < self.PosList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers
    def get_finger_coordinates(self, img):
        x, y, w, h = 0, 0, 0, 0

        # Find hands and get finger positions
        img = self.findHands(img)
        lm_list = self.findPosition(img, draw=False)
        finger_up = self.fingerUp()

        if len(lm_list) != 0:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]

            # Drawing mode - Index finger is up
            if finger_up[1] and not finger_up[2]:
                x, y, w, h = x1, y1, abs(x2 - x1), abs(y2 - y1)

        return x, y, w, h



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(detectionCon=0.75)
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
