import cv2
import time
import numpy as np
import os
import HandTrackingModule as htm
from tensorflow.keras.models import load_model
#######################
brushThickness = 15
eraserThickness = 50
#######################
pTime = 0
cTime = 0

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.75)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

model = load_model('digit_recognition_model.h5')

while True:
    # 1. import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3. check which fingers are up
        finger=detector.fingerUp()
        #print(finger)
        # 4. if selection mode - two fingers are up
        if finger[1] and finger[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # # checking for the click
            if y1 < 125:
                     if 250 < x1 < 450:
                         header = overlayList[0]
                         drawColor = (0, 0, 255)
                     elif 550 < x1 < 750:
                         header = overlayList[1]
                         drawColor = (255, 0, 0)
                     elif 800 < x1 < 950:
                         header = overlayList[2]
                         drawColor = (0, 255, 0)
                     elif 1050 < x1 < 1200:
                         header = overlayList[3]
                         drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        # 5. if drawing mode - index finger is up
        if finger[1] and finger[2]==False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp=x1,y1

            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor
                            ,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)

            xp,yp=x1,y1
    x, y, w, h = detector.get_finger_coordinates(img)

    # Check if the bounding box is valid
    if w > 0 and h > 0:
        # Extract the region where the user draws the digit
        digit_roi = img[y:y + h, x:x + w]

        # Check if digit_roi is not empty
        if digit_roi.size != 0:
            # Preprocess the drawn digit image
            processed_digit = cv2.cvtColor(digit_roi, cv2.COLOR_BGR2GRAY)
            processed_digit = cv2.resize(processed_digit, (28, 28))
            processed_digit = processed_digit.astype('float32') / 255.0
            processed_digit = np.expand_dims(processed_digit, axis=0)

            # Use the trained CNN model to predict the digit
            prediction = model.predict(processed_digit)
            predicted_digit = np.argmax(prediction)

            # Display the predicted digit on the screen
            cv2.putText(img, f'Prediction: {predicted_digit}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (100, 700), cv2.FONT_HERSHEY_PLAIN, 3, (0, 165, 255), 3)
    #setting the header image
    img[0: 125, 0: 1280] = header
    #img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('image.jpg', img)
        break
cv2.destroyAllWindows()
