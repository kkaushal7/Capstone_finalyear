from __future__ import division

import tensorflow as tf

import tensornets as nets
import cv2
import numpy as np
import time
import logging
import logging.config
import time

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

logging.config.fileConfig('logging.ini')
tf.compat.v1.disable_eager_execution()


VIDEO_PATH = 'testdata/sample_video_1_crop.mp4'
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame
classes={'2':'car'}
list_of_classes=[2]
not_same_cones = 0
no_of_crashes = 0

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 20  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5
CRASH_THRESHOLD = 0.7
CAR_SCORE_THRESHOLD = 0.7
            
def iou(box1, box2,frame):
    global no_of_crashes
    is_crash = False
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = xi2-xi1
    inter_height = yi2-yi1
    inter_area = max(inter_width,0)*max(inter_height,0)
    box1_area = (box1_x1-box1_x2)*(box1_y1-box1_y2)
    box2_area = (box2_x1-box2_x2)*(box2_y1-box2_y2)
    union_area = box1_area+box2_area-inter_area
    iou = max(inter_area/union_area*box2_area/union_area,0)
    iou1 = max(inter_area/union_area*box2_area/union_area,0)
    iou2 = max(inter_area/union_area*box2_area/box1_area,0)
    
##    if(iou>0):
##        print(box1,box2,iou,iou1,iou2)
##        print(inter_area,union_area)
##        cv2.rectangle(frame,(xi1,yi1),(xi2,yi2),(255,255,0),3)
    
    
    if(iou2 > CRASH_THRESHOLD):
        print(box1,box2,iou,iou1,iou2)
        print(inter_area,union_area)
        cv2.rectangle(frame,(xi1,yi1),(xi2,yi2),(255,255,0),3)
        no_of_crashes += 1
        return not is_crash
    
    
    return is_crash


def check_accidents(count,count_prev,cones,cars,no_of_cones,frame):
    is_crash = False
    global not_same_cones
    if count == 0 and count_prev >0:
        if(len(cones) <= no_of_cones):
            is_crash = True
            not_same_cones = max(not_same_cones,(no_of_cones - len(cones)))
    

    print("Cones : "+str(len(cones))+" Cars : "+str(len(cars)))        
            
    if len(cones) ==0 or len(cars)==0:
        return is_crash

        
    if count > 0:
        for i, (cone,car) in enumerate(zip(cones,cars)):
            is_crash = iou(cone,car,frame)
            if is_crash:
                return is_crash
            
        
    return is_crash


    
    
def find_crash(no_of_cones):
    
    count_prev = 0
    global no_of_crashes 
    
    #load graphs
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)
    sess_tc = tf.compat.v1.Session(graph=detection_graph)
    
    default_graph = tf.compat.v1.get_default_graph()
    with default_graph.as_default():
        inputs = tf.compat.v1.placeholder(tf.float32, [None, 416, 416, 3]) 
        model = nets.YOLOv3COCO(inputs, nets.Darknet19)
    
     

    #load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    processed_images = 0
    
    #sessions for running graphs 
    
    sess = tf.compat.v1.Session(graph = default_graph)
    sess.run(model.pretrained())
    
    while(cap.isOpened()):
        
        if DETECT_EVERY_N_SECONDS:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        processed_images * fps * DETECT_EVERY_N_SECONDS)
                
        ret, frame = cap.read()

        cones = []
        cars = []
        if ret:
            
            start_time=time.time()
            
            #resizing for yolo
            img=frame
            imge=np.array(img).reshape(-1,frame.shape[0],frame.shape[1],3)

            #count cars
            preds = sess.run(model.preds, {inputs: model.preprocess(imge)})
            boxes_car = model.get_boxes(preds, imge.shape[1:3])
            #plotting boxes_cars
            boxes1=np.array(boxes_car)
            for j in list_of_classes: #iterate over classes
                count =0
                if str(j) in classes:
                    lab=classes[str(j)]
                if len(boxes1) !=0:
                #iterate over detected vehicles
                    for i in range(len(boxes1[j])): 
                        box=boxes1[j][i]
                        box = box[0:4]
                        #setting confidence threshold as 40%
                        if boxes1[j][i][4]>=CAR_SCORE_THRESHOLD: 
                            count += 1
                            cars.append(box)
                            cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(0,255,255),3)
                            cv2.putText(frame, lab, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)

                
            if (count_prev>0 or count>0):
                #Extracting crops & resizing factor for traffic cone
                crops, crops_coordinates = ops.extract_crops(
                        frame, CROP_HEIGHT, CROP_WIDTH,
                        CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

                #perdicting cones
                detection_dict = tf_utils.run_inference_for_batch(crops, sess_tc,detection_graph)

                #plotting cones cordinates
                # (ymin, xmin, ymax, xmax)
                boxes = []
                for box_absolute, boxes_relative in zip(crops_coordinates, detection_dict['detection_boxes']):
                    boxes.extend(ops.get_absolute_boxes(box_absolute,boxes_relative[np.any(boxes_relative, axis=1)]))
                if boxes:
                    boxes = np.vstack(boxes)

                # Remove overlapping boxes
                boxes = ops.non_max_suppression_fast(
                    boxes, NON_MAX_SUPPRESSION_THRESHOLD)

                # Get scores to display them on top of each detection
                boxes_scores = detection_dict['detection_scores']
                boxes_scores = boxes_scores[np.nonzero(boxes_scores)]
                

                for box, score in zip(boxes, boxes_scores):
                    if score > SCORE_THRESHOLD:
                        ymin, xmin, ymax, xmax = box
                        box1 = [xmin,ymin,xmax,ymax]
                        cones.append(box1)
                        image = cv2.circle(frame, (xmin,ymin), 5, (255,255,0),3) 
                        cv2.rectangle(frame,(box1[0],box1[1]),(box1[2],box1[3]),(0,255,0),3)
                        #color_detected_rgb = cv_utils.predominant_rgb_color(frame, ymin, xmin, ymax, xmax)
                        #text = '{:.2f}'.format(score)
                        #cv_utils.add_rectangle_with_text(frame, ymin, xmin, ymax, xmax,color_detected_rgb, text)

                is_crash = check_accidents(count,count_prev,cones,cars,no_of_cones,frame)
                if(is_crash):
                    print(is_crash,no_of_crashes,not_same_cones)
                #logging.debug('Detected {} objects in {} images in {:.2f} ms'.format(len(boxes), len(crops), processing_time_ms))
                

            count_prev = count
            if OUTPUT_WINDOW_WIDTH:
                    frame = cv_utils.resize_width_keeping_aspect_ratio(
                        frame, OUTPUT_WINDOW_WIDTH)

            cv2.imshow('Detection result', frame)
            cv2.waitKey(1)
        
            
            processed_images += 1

            end_time = time.time()
            processing_time_ms = (end_time - start_time) * 1000


        else:
            # No more frames. Break the loop
            break
        

    #closing sessions    
    cap.release()
    cv2.destroyAllWindows()
    sess.close()
    sess_tc.close()
    

    
def main():
    initial_cones = 3
    global not_same_cones
    global no_of_crashes
    not_same_cones = 0
    no_of_crashes = 0
    find_crash(initial_cones)
    no_of_crashes = min(initial_cones,max(no_of_crashes,not_same_cones))
    print("No of Crashes : "+str(no_of_crashes))
    
if __name__ == '__main__':
    main()
