import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

import uuid
import os
import time


#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


#Image Detection
# img = "https://media.cntraveler.com/photos/53e2f41cdddaa35c30f66775/1:1/w_960,c_limit/highway-traffic.jpg"
# result = model(img)
# result.show()
# result.print()

#Live Object detection
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()

#     #Make detection
#     results = model(frame)

#     cv2.imshow('Dorwsiness Detection', np.squeeze(results.render()))

#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


####################################################################
#Create Images For Model training

# IMAGES_PATH = os.path.join('data','images')
# labels = ['awake', 'drowsy']
# number_imgs = 20


# cap = cv2.VideoCapture(0)

# #loop for lables
# for label in labels:
#     print('Collecting images for {}'.format(label))
#     time.sleep(5)

#     #loop through images
#     for img_num in range(number_imgs):
#         print('Collecting inmages for {}, image number {}'.format(label,img_num))
        
#         #webcame feed
#         ret, frame = cap.read()
        
#         #nameing the images
#         imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
       
#         #write image to file
#         cv2.imwrite(imgname, frame)

#         #render to screen
#         cv2.imshow('Image Collection', frame)

#         # 2 second delay between captures
#         time.sleep(2)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()

#######################################################################################
model = torch.hub.load('ultralytics/yolov5','custom',path='yolov5/runs/train/exp2/weights/last.pt', force_reload=True)

# img = os.path.join('data','images','awake.1d82ffb9-e46c-11ed-81d2-b4e1aff3a71a.jpg')
# results = model(img)
# results.print()
# results.show()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    #Make detection
    results = model(frame)

    cv2.imshow('Dorwsiness Detection', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
