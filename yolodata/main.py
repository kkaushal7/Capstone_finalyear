import os
import cv2
import numpy as np



net = cv2.dnn.readNet("F:\\Internet Download Manager\\yolov3_final.weights","F:\\Internet Download Manager\\yolov3.cfg")
classes = []
with open("G:\\AI\\fifa18\\Project2-submission\\Lane-Lines-Detection-Python-OpenCV-master\\zebra crossing\\carobjectdetection\\crosswalk-regulator-core-master\\obj.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
violations=False


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1);print(interArea)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = float(interArea) / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def detect(frame):
    global violations
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = frame
    cimg = img
    # cimg = cv2.resize(img, (720,720)) 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape
    # print size

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    foundred = False
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                foundred = True
    if foundred:
        # img1 = cv2.resize(cimg, (416,416)) 
        height,width,channels = cimg.shape
        blob = cv2.dnn.blobFromImage(cimg,0.00392,(416,416),(0,0,0),True,crop=False)
        net.setInput(blob)
        outs = net.forward(outputlayers)
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
        carbox=[]
        walkbox=[]
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                if(label == 'car'):
                    carbox.append((x,y,w,h))
                elif(label == 'crosswalk'):
                    walkbox.append((x,y,w,h))
                cv2.rectangle(cimg,(x,y),(x+w,y+h),(255, 0, 0),2)
                cv2.putText(cimg,label,(x,y+30),font,1,(255,255,255),2)
        if(len(carbox) > 0 and len(walkbox) >0):
            for carb in carbox:
                x1,y1,w1,h1 = carb
                x2,y2,w2,h2 = walkbox[0]
                # print(x2,x2+w2)
                # print(x1,x1+w1)
                # print(y2,y2+h2)
                # print(y1,y1+h1)

            # left = max(x1, x2)
            # right = min(x1+w1, x2+w2)
            # bottom = min(y1+h1, y2+h2)
            # top = max(y1, y2)
            # if left < right and bottom > top and ((y1+h1)/2 < (y2+h2) or  (y1+h1) < (y2+h2)):
            # iou = bb_intersection_over_union(carbox[0], walkbox[0])
                dx = min(x1+w1, x2+w2) - max(x1, x2)
                dy = min(y1+h1, y2+h2) - max(y1, y2)
                if (2*y1+h1)/2 < (y2+h2) and (x1+w1) <= (x2+w2):
                    if (dx>=0) and (dy>=0):
                        violations = True
                        break
            
                # print(x1,x1+w1)
            # if(x1+w1 <= x2+w2):
                # violations = True
            # print(violations)
            # if (iou>=1.8):
            #     print(iou)
            #     violations = True
            #     # print((right-left)*(bottom-top))
               
                

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                
            

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                

    cv2.imshow('detected results', cimg)
    # cv2.imshow("Image",img1)
    
    
if cv2.waitKey(1)&0xFF==ord('q'):   # To get the correct frame rate
    cv2.destroyAllWindows()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


if __name__ == '__main__':

    # cap=cv2.VideoCapture('http://172.31.65.50:8081/video.cgi?.mjpg')
    cap=cv2.VideoCapture('G:\\AI\\fifa18\\Project2-submission\\Lane-Lines-Detection-Python-OpenCV-master\\zebra crossing\\carobjectdetection\\aziz1.MP4')
    font = cv2.FONT_HERSHEY_COMPLEX
    while cap.isOpened():
        success, frame = cap.read()
        if success is False:
            break
        # initial_image = frame
        initial_image = cv2.resize(frame, (720,720)) 
        detect(initial_image)
        if cv2.waitKey(1)&0xFF==ord('q'):   
            break
    cap.release()
    print("Violation ",violations)

    if violations == True:
        print("Immediate Fail")
    else:
        print("Passed")
        
        


