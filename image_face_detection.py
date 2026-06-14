#!/usr/bin/env python
# coding: utf-8

# In[63]:


import cv2 
import matplotlib.pyplot as plt 
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[64]:


caffeModel=model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config_file = "models/deploy.prototxt"

net=cv2.dnn.readNetFromCaffe(config_file,caffeModel)

im=r"C:\Users\91850\Pictures\1.JPG" #Image Path 
img=cv2.imread(im)
plt.figure(figsize=[10,5])
plt.imshow(img[:,:,::-1])
plt.axis('off')


# In[67]:


def detect_face(image,detection_threshold=0.7):
    blob=cv2.dnn.blobFromImage(image,1.0,(300,300),[104,117,123])
    net.setInput(blob)
    detections=net.forward()
    faces=[]
    img_h=image.shape[0]
    img_w=image.shape[1]
    
    for  detection in detections[0][0]:
        if detection[2]>=detection_threshold:
            left   =detection[3]*img_w
            top    =detection[4]*img_h
            right  =detection[5]*img_w
            bottom =detection[6]*img_h
            face_w=right-left
            face_h=bottom-top
            face_roi=(left,top,face_w,face_h)
            faces.append(face_roi)

    return np.array(faces).astype(int)
    


# In[77]:


faces=detect_face(img)
img_display=img.copy()
for face in faces:
    cv2.rectangle(img_display,face,(0,225,0),18)
plt.imshow(img_display[:,:,::-1])


# In[78]:





# In[ ]:




