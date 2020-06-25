
# @Description - Lane detection techniques


import cv2
import numpy as np
import vector
import urllib.request 
import matplotlib.pyplot as plt
from numpy.linalg import inv
import argparse
from os import path
from time import sleep
from utm import to_latlon

import CameraParams_Parser as Cam_Parser
import GPS_VO
import cv2
import Ground_Truth as GT
import Trajectory_Tools as TT
from Common_Modules import *
from py_MVO import VisualOdometry


from filesream import FileVideoStream
import time

# GLobal params
nwindows = 10
margin=110 #width offset line side ways
minpix=20


################
scoreval = []
################

counterrors = 0

#leftlane = 0
#rightlane = 1

currentlane = -1
previouslane = -1

def getImages(video):
	fvs = FileVideoStream(video).start()
	time.sleep(1.0)
	i=0
	while fvs.more():
		frame = fvs.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.fastNlMeansDenoising(gray)
		frame = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)
		padded_img = cv2.copyMakeBorder(
    	frame,0,0,0,0,borderType=cv2.BORDER_CONSTANT,value=[0,0,0])
		cv2.imwrite("images/"+str(i).zfill(6)+'.png',padded_img)
		
		i+=1
	fvs.stop()


width=1280
height = 720
dim = (width, height)
def run():
    # getImages("G:\\AI\\attachments\\Advanced-Lane-Detection-master\\challenge_video.mp4")
    print ('-- Press ESC key to end program\n')
    #Parse the Command Line/Terminal
    # cmd_parser = argparse.ArgumentParser()
    # cmd_parser.add_argument('txt_file', help= 'Text file that contains all the input parameters. Verify the CameraParams file.')
    # args = cmd_parser.parse_args()

    CP = Cam_Parser.CameraParams('CameraParams.txt')

    # Returns the images' directory, images' format, list of images and GPS_FLAG
    folder, img_format, images, GPS_flag = CP.folder, CP.format, CP.images,CP.GPS_FLAG
    
    gps_switch = False  # Determines if GPS was recovered
    if GPS_flag == 'GPS_T' or GPS_flag == 'GPS_T_M':  # Verify if flag was raised
        gps_dict = GPS_VO.gps_filename_dict(images)  # Retrieve the GPS info into a dictionary
        utm_dict = GPS_VO.gps_to_utm(gps_dict)       # Keys: image filepath Values: GPS coordinates
        if gps_dict and utm_dict:
            gps_switch = True  # Verify if GPS info was retrieved
            # Write GPS to text file in the images sequences directory
            GPS_utm_coord = open(path.normpath(folder + '/raw_GPS.txt'), 'w')
            for key, value in utm_dict.items():
                value = to_latlon(value[0], value[1], 17, 'U') # Specific to Pittsburgh
                GPS_utm_coord.write(key + ' ' + str(value[0]) + ' ' + str(value[1]) + '\n')
            GPS_utm_coord.close()  # Close the Poses text file
        else:
            print ("Warning: No GPS data recovered from the images' EXIF file")

    # Returns the camera intrinsic matrix, feature detector, ground truth (if provided), and windowed displays flag
    K, f_detector, GT_poses, window_flag = CP.CamIntrinMat, CP.featureDetector, CP.groundTruth, CP.windowDisplay
    # Initializing the Visual Odometry object
    vo = VisualOdometry(K, f_detector, GT_poses)
    # Square for the real-time trajectory window
    traj = np.zeros((1280, 720, 3), dtype=np.uint8)

    # ------------------ Image Sequence Iteration and Processing ---------------------
    # Gives each image an id number based position in images list
    img_id = 0
	
    T_v_dict = OrderedDict()  # dictionary with image and translation vector as value
    # Initial call to print 0% progress bar
    TT.printProgress(img_id, len(images)-1, prefix='Progress:', suffix='Complete', barLength=50)
	
	

    for i, img_path in enumerate(images):  # Iterating through all images
        print(img_path);print(i)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # Wait for ESC key to exit
            cv2.destroyAllWindows()
            return
        # Read the image for Visual Odometry
        imgKLT = cv2.imread(img_path)  
        img = cv2.imread(img_path, 0);img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		
        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        img = clahe.apply(img)

        if img_id != 0 and gps_switch is True:
            # Retrieve image distance in order to scale translation vectors
            prev_GPS = gps_dict.values()[img_id - 1]
            cur_GPS = gps_dict.values()[img_id]
            distance = GPS_VO.getGPS_distance(prev_GPS, cur_GPS)  # Returns the distance between current and last GPS

        if vo.update(img, img_id):  # Updating the vectors in VisualOdometry class
            if img_id == 0:
                T_v_dict[img_path] = ([[0], [0], [0]])
            else:
                T_v_dict[img_path] = vo.cur_t   # Retrieve the translation vectors for dictionary
            cur_t = vo.cur_t  # Retrieve the translation vectors

            # ------- Windowed Displays ---------
            if window_flag == 'WINDOW_YES':
                if img_id > 0:  # Set the points for the real-time trajectory window
                    x, y, z = cur_t[0], cur_t[1], cur_t[2]
                    TT.drawOpticalFlowField(imgKLT, vo.OFF_prev, vo.OFF_cur)  # Draw the features that were matched
                else:
                    x, y, z = 0., 0., 0.

                traj = TT.RT_trajectory_window(traj, x, y, z, img_id)  # Draw the trajectory window
            # -------------------------------------
        sleep(0.1) # Sleep for progress bar update
        img_id += 1  # Increasing the image id
        TT.printProgress(i, len(images)-1, prefix='Progress:', suffix='Complete', barLength=50)  # update progress bar

    # --------------------------------------------------------------------------------

    # Write poses to text file in the images sequences directory
    poses_MVO = open(path.normpath(folder+'/py-MVO_Poses.txt'), 'w')
    for t_v, R_m in zip(vo.T_vectors, vo.R_matrices):
        T = np.hstack((R_m, t_v)).flatten()
        poses_MVO.write(' '.join([str(t) for t in T]) + '\n')
    poses_MVO.close()  # Close the Poses text file

    # Write the images path and translation vector to text file in the images sequences directory
    VO_t = open(path.normpath(folder + '/py-MVO_TV.txt'), 'w')
    # Retrieving the translation vectors from the
    # translation vector dictionary and write it in a txt file
    T_v = []
    for key, value in T_v_dict.items():
        T_v_dict[key] = np.array((value[0][0], value[2][0]))
        T_v.append((value[0][0], value[2][0]))
        VO_t.write(key + ' ' + str(value[0][0]) + ' ' + str(value[2][0]) + '\n')

    VO_t.close()  # Close the Poses text file

    # -------- Plotting Trajectories ----------
    if window_flag == 'WINDOW_YES' or window_flag == 'WINDOW_T':

        if GT_poses:  # Ground Truth Data is used in case GPS and GT are available
            # Ground Truth poses in list
            GT_poses = GT.ground_truth(GT_poses)
            # Plot VO and ground truth trajectories
            TT.VO_GT_plot(T_v, GT_poses)

        elif gps_switch:  # Plotting the VO and GPS trajectories
            if GPS_flag == 'GPS_T':
                TT.GPS_VO_plot(T_v, utm_dict)
            elif GPS_flag == 'GPS_T_M':
                # Do merged trajectory
                VO_dict = TT.GPS_VO_Merge_plot(T_v_dict, utm_dict)

                # Write GPS to text file in the images sequences directory
                VO_utm_coord = open(path.normpath(folder + '/py-MVO_GPS.txt'), 'w')
                for key, value in VO_dict.items():
                    value = to_latlon(value[0], value[1], 17, 'U')
                    VO_utm_coord.write(key+' '+str(value[0])+' '+str(value[1])+'\n')
                VO_utm_coord.close()  # Close the Poses text file

        else:
            TT.VO_plot(T_v)
    # -------------------------------------------

    return




def correct_dist(initial_img):
	# Intrnsic camera matrix
	k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
		 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	k = np.array(k)
	# Distortion Matrix
	dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
	dist = np.array(dist)
	img_2 = cv2.undistort(initial_img, k, dist, None, k)

	return img_2




# Turn prediction for lanes
def turn_predict(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    
    if (lane_center - image_center < 0):
        return ("Turning left")
    elif (lane_center - image_center < 8):
        return ("straight")
    else:
    	return ("Turning right")


# Process image and homography operations
def image_preprocessing(img):
	global counterrors
	global currentlane
	global previouslane
	# print("$$$$")

	crop_img = img[420:720, 40:1280, :]  # To get the region of interest
	
	undist_img = correct_dist(crop_img)

	hsl_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HLS)

	# To seperate out Yellow colored lanes
	lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
	upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
	mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)

	yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

	# To seperate out White colored lanes
	lower_mask_white = np.array([0, 200, 0], dtype='uint8')
	upper_mask_white = np.array([255, 255, 255], dtype='uint8')
	mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)

	white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

	# Combine both
	lanes = cv2.bitwise_or(yellow_detect, white_detect)

	new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)

	final = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)

	# Filter noise
	img_blur = cv2.bilateralFilter(final, 9, 120, 100)

	# Apply edge detection
	img_edge = cv2.Canny(img_blur, 100, 200)


	##############
	# indices = np.where(img_edge != [0])
	# coordinates = zip(indices[0], indices[1])

	# histo = np.sum(img_edge, axis=0)
	# midly = np.int(histo.shape[0]/2)
	# rigx_ = np.argmax(histo[midly:]) + midly
	# lefx_ = np.argmax(histo[:midly])
	# print("lefx_ ",lefx_)
	# print("rigx_ ",rigx_)

	##############
	
    
	# Apply homography to get bird's view
	new_img = cv2.warpPerspective(img_edge, H_matrix, (300, 600))

	# new_img1 = cv2.warpPerspective(img_edge, H_matrix, (720, 600))
	
	# Use histogram to get pixels with max Y axis value
	histogram = np.sum(new_img, axis=0)
	out_img = np.dstack((new_img,new_img,new_img))*255
	
	midpoint = np.int(histogram.shape[0]/2)
	
	# Compute the left and right max pixels
	leftx_ = np.argmax(histogram[:midpoint])
	rightx_ = np.argmax(histogram[midpoint:]) + midpoint
	# print(leftx_)
	
	left_lane_pos = leftx_
	right_lane_pos = rightx_
	image_center = int(new_img.shape[1]/2)
	
	

	# image_center = img_edge.shape[0]/2
	# Use the lane pixels to predict the turn
	prediction = turn_predict(image_center, right_lane_pos, left_lane_pos)	
			
	window_height = np.int(new_img.shape[0]/nwindows)

	nonzero = new_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	# Update current position for each window
	leftx_p = leftx_
	rightx_p = rightx_
	
	# left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_down = new_img.shape[0] - (window+1)*window_height
		win_y_up = new_img.shape[0] - window*window_height
		win_x_left_down = leftx_p - margin
		win_x_left_up = leftx_p + margin
		win_x_right_down = rightx_p - margin
		win_x_right_up = rightx_p + margin
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
		# print(good_left_inds)
		good_right_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]
		
		# Append these indices to the list
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If found > minpix pixels, move to next window
		if len(good_left_inds) > minpix:
			leftx_p = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_p = np.int(np.mean(nonzerox[good_right_inds]))
	
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 


	if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
		return

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 255, 255]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 255, 255]
	
	
	# Fit a second order polynomial to each
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# Extract points from fit
	left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                              ploty])))])

	image_center = img_edge.shape[0]/2

	
	pts = np.hstack((left_line_pts, right_line_pts))
	pts = np.array(pts, dtype=np.int32)
    
	# print(left_lane_pos)
	# print(right_lane_pos)
	
	# print("##1")
	# print(leftx)
	
	# print(rightx)
	
	# if (left_lane_pos>=min(leftx) and left_lane_pos<=max(leftx) and right_lane_pos<= min(rightx) and prediction == "Turning right" ):
	# 	counterrors = counterrors + 1
	# 	print(prediction)
	# 	print("$$")
	# 	print(left_lane_pos)
	# 	print(right_lane_pos)
	# 	print("##1")
	# 	print(leftx)
	# 	print(rightx)
	# 	print("$$")
	# if(prediction == "Turning right" or prediction == "straight"):
	# 	if (left_lane_pos<min(leftx) or left_lane_pos>max(leftx) or right_lane_pos< min(rightx) or right_lane_pos>max(rightx)):
	# 		print("$$")
	# 		print(left_lane_pos)
	# 		print(right_lane_pos)
	# 		print("##1")
	# 		print(leftx)
	# 		print(rightx)
	# 		print("$$")
	# 		counterrors = counterrors + 1
	 
	

	# if(prediction == "Turning left") and (left_line_pts - left_lane_pos)<=0 :
	# 	print("true")

	color_blend = np.zeros_like(img).astype(np.uint8)
	cv2.fillPoly(color_blend, pts, (0,255, 0))
	
	# Project the image back to the orignal coordinates
	newwarp = cv2.warpPerspective(color_blend, inv(H_matrix), (crop_img.shape[1], crop_img.shape[0]))


###############333
	# newwarp1 = cv2.warpPerspective(out_img, inv(H_matrix), (crop_img.shape[1], crop_img.shape[0]))

	# newwarp2 = cv2.warpPerspective(new_img1, inv(H_matrix), (crop_img.shape[1], crop_img.shape[0]))
###################
	
	# out_img = cv2.Canny(out_img, 180, 200)
	out_img1 = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
	_,contours,_ = cv2.findContours(out_img1, 1, 2)
	# imgi = cv2.drawContours(out_img1, contours, -1, 255, 3)s
	c = max(contours, key = cv2.contourArea)
	x,y,w,h = cv2.boundingRect(c)
	imgi = cv2.rectangle(out_img1,(x,y),(x+w,y+h),(255,0,0),2)
	# cv2.imshow('change',out_img1)

	image_center1 = out_img1.shape[1]/2

	if(previouslane == -1):
		if(x < image_center1):
			currentlane = 0
		elif(x > image_center1):
			currentlane = 1
	elif(previouslane == 0):
		if(x>image_center1 and prediction !="Turning right"):
			currentlane=1
			counterrors = counterrors + 1
		elif x>image_center1:
			currentlane = 1
			# print(counterrors)
	else:
		if(x<image_center1 and prediction !="Turning left"):
			currentlane=0
			counterrors = counterrors + 1
		elif x<image_center1:
			currentlane=0
			# print(counterrors)
	previouslane = currentlane
	# print("currentlane ",currentlane)




    ###$$$$$######

	# print(crop_img.shape)
	result = cv2.addWeighted(crop_img, 2, newwarp, 0.5, 0)
	cv2.putText(result, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv2.LINE_AA)

	# Show the output image
	cv2.imshow('result-image',result)



def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# Source points for homography
src = np.array([[500, 50], [686, 41], [1078, 253], [231, 259]], dtype="float32")

# Destination points for homography
dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]], dtype="float32")

#Homography 
H_matrix = cv2.getPerspectiveTransform(src, dst)
Hinv = inv(H_matrix)


# Read the frames
traj = np.zeros((600, 600, 3), dtype=np.uint8)
pspeed = 0.0
pxcord = 0.0
pycord = 0.0

x_vals=[]
y_vals=[]


fvs = FileVideoStream('221.mp4').start()
time.sleep(1.0)





while fvs.more():
# 	# fvs.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
	frame = fvs.read()
	# print(frame.shape)
# 	# initial_image = rescale_frame(frame, percent=150)
# 	# cv2.imshow('result-image',initial_image)
	# print(initial_image.shape)
	width = 1280
	height = 720
	dim = (width, height)
	resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
	image_preprocessing(resized)
	
	#for smartphone behaviour analysis score
	stream = urllib.request.urlopen('http://192.168.1.5:8081/frame.mjpg')
    res = str(stream.info()).split()
	scoreval.append(float(res[-1]))
    #######

	
	

	
	
	
	if cv2.waitKey(1)&0xFF==ord('q'):   
		break

cv2.destroyAllWindows()
fvs.stop()



print('counterrors: ' + str(counterrors))


penalty = 0.1*counterrors
finalscore = 0.0
if len(scoreval) > 0:
finalscore = float(sum(scoreval)) / float(len(scoreval))

print(finalscore)



# getImages("challenge_video.mp4")
# run()