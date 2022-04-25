# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 09:45:47 2021

@author: Azhar Rizky Zulma
"""

import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
#import os

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 640)
#cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
#fpsReader = cvzone.FPS()

imgBg = cv2.imread("bromo.jpg")
imgBg = imgBg[:480, :640, :]

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgBg, threshold=0.8)
    
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    #_, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))
    
    cv2.imshow("After Optimization", imgStacked)
    cv2.waitKey(1)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release() # Releases webcam or capture device
cv2.destroyAllWindows() # Closes imshow frames
