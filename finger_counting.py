import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

confidence = [0,0,0,0,0,0]
confidence_max=10
conf_i=0
max_i=0

def nothing(x):
    pass

cv2.namedWindow('sliders')
cv2.namedWindow('sliders2')

cv2.createTrackbar('h0', 'sliders',0,255,nothing)
cv2.createTrackbar('h1', 'sliders',38,255,nothing)
cv2.createTrackbar('s0', 'sliders',34,255,nothing)
cv2.createTrackbar('s1', 'sliders',255,255,nothing)
cv2.createTrackbar('v0', 'sliders',0,255,nothing)
cv2.createTrackbar('v1', 'sliders',151,255,nothing)

cv2.createTrackbar('x', 'sliders2',50,640,nothing)
cv2.createTrackbar('y', 'sliders2',50,480,nothing)
cv2.createTrackbar('size', 'sliders2',300,480,nothing)

while(1):
    ret, frame = cap.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100) # Smoothing
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)

    h0= cv2.getTrackbarPos('h0','sliders')
    h1= cv2.getTrackbarPos('h1','sliders')
    s0= cv2.getTrackbarPos('s0','sliders')
    s1= cv2.getTrackbarPos('s1','sliders')
    v0= cv2.getTrackbarPos('v0','sliders')
    v1= cv2.getTrackbarPos('v1','sliders')
    x= cv2.getTrackbarPos('x','sliders2')
    y= cv2.getTrackbarPos('y','sliders2')
    size= cv2.getTrackbarPos('size','sliders2')

    #define region of interest
    roi=frame[x:(x+size), y:(y+size)]

    cv2.rectangle(frame,(y,x),(y+size,x+size),(0,255,0),0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # define range of skin color in HSV

    lower_skin = np.array([h0,s0,v0], dtype=np.uint8)
    upper_skin = np.array([h1,s1,v1], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask,kernel,iterations = 2)
    mask = cv2.dilate(mask,kernel,iterations = 4)
    mask = cv2.GaussianBlur(mask,(5,5),100)

    #find contours
    contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        continue

    #find contour of max area(hand)
    cnt = max(contours, key = lambda x: cv2.contourArea(x))

    #approx the contour a little
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)

    #make convex hull around hand
    hull = cv2.convexHull(cnt)

    #find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    if defects is None:
        cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        continue

    # l = no. of defects
    l=0

    #code for finding no. of defects due to fingers
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)
        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90:
            l += 1
            cv2.circle(roi, far, 3, [255,0,0], -1)

        #draw lines around hand
        cv2.line(roi,start, end, [0,255,0], 2)

    l+=1
    font = cv2.FONT_HERSHEY_SIMPLEX

    if max_i==1:
        cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif max_i==2:
        cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif max_i==3:
        cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif max_i==4:
        cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    #else :
    # cv2.putText(frame,'Out of range',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    if l>5:
        l=5

    #show the windows
    cv2.imshow('mask',mask)
    cv2.imshow('frame',frame)

    confidence[l-1] = confidence[l-1]+1
    conf_i=conf_i+1
    if conf_i==confidence_max:
        conf_i=0
        max_i=0
        for i in range(0,5):
            if confidence[i]>confidence[max_i]:
                max_i=i
        max_i+=1
        confidence = [0,0,0,0,0,0]
    print("Fingers=",max_i)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord(' '):
        while True:
            k1 = cv2.waitKey(10) & 0xFF ;
            if k1 == ord(' '):
                break
            elif ( k1 == ord('s') ) | ( k1 == ord('S') ):
                s=str(max_i)+"\n"
                print(s)
cv2.destroyAllWindows()
cap.release()