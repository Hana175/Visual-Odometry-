"""

MIT License (MIT)

Copyright (c) SUMMER 2016, Carnegie Mellon University

Author: Jahdiel Alvarez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Original Code:
https://github.com/uoip/monoVO-python


"""
import warnings
import argparse
import ast
import csv
from os import listdir, path
import os
import readline
import resource
import numpy as np
import time
from time import sleep
from cv2 import CV_8UC1, imread
#comment el ta7ty de ay 7aga pil and matplot lib
from matplotlib.animation import ImageMagickBase
from utm import to_latlon
from PIL import Image
import glob
import collections
import CameraParams_Parser as Cam_Parser
import GPS_VO
#import Ground_Truth as GT
import Trajectory_Tools as TT
from Common_Modules import *
from py_MVO import VisualOdometry
import glob, os
import datetime
import imutils
from imutils.video import VideoStream
import py_MVO as PY



def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def run():
    vidcap = cv2.VideoCapture('/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bike/output.avi')
    success,image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_id = 0
    path = "/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikeframes/"
    # for f in os.listdir(path):
    #     os.remove(os.path.join(path, f))
    # while success:
    #     cv2.imwrite(os.path.join(path ,"frame%d.png" % img_id), image)     # save frame as PNG file      
    #     success,image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     img_id += 1

    time_start = time.perf_counter()

    # print('-- Press ESC key to end program\n')
    # #Parse the Command Line/Terminal
    txt_file = "/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/CameraParams.txt"
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('txt_file', help= 'Text file that contains all the input parameters. Verify the CameraParams file.')
    args = cmd_parser.parse_args()
    #print(args.txt_file)
    CP = Cam_Parser.CameraParams(args.txt_file)
    img_path = "/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikeframes/"

    # Returns the images' directory, images' format, list of images and GPS_FLAG
    folder, images, GPS_flag = CP.folder, CP.images, CP.GPS_FLAG

    # Returns the camera intrinsic matrix, feature detector, ground truth (if provided), and windowed displays flag
    K, f_detector, window_flag = CP.CamIntrinMat, CP.featureDetector, CP.windowDisplay
    
    K[0,0]= K[0,0]*0.75
    K[1,1]= K[1,1]*0.75
    K[0,2]= K[0,2]*0.75
    K[1,2]= K[1,2]*0.75
    # Initializing the Visual Odometry object
    vo = VisualOdometry(K, f_detector)
    # Square for the real-time trajectory window
    #traj = np.zeros((600, 600, 3), dtype=np.uint8)

    # ------------------ Image Sequence Iteration and Processing ---------------------
    # Gives each image an id number based position in images list
    T_v_dict = OrderedDict()  # dictionary with image and translation vector as value
    
   
    
    os.chdir("/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikeframes/")
    path = "/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikeframes/"
    
    count = 0
    for file in os.listdir(path):
       # print(file)
        
    
   
        img_path = path+file
        
        time_elapsed = (time.perf_counter() - time_start)
        memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1014.0/1014.0
        print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb)) # to calculate memory and computational time
        print("fps= ", count/time_elapsed)
        
        
        #imgKLT = cv2.imread(img_path,0)  # Read the image for real-time trajectory
        img = cv2.imread(img_path, 3)  # Read the image for Visual Odometry
        width = int(img.shape[1]*1.0)
        height = int(img.shape[0]*1.0)
        dim = (width, height)
        
        img = cv2.resize(img, dim)
        print(img.shape)
        #imgKLT = cv2.resize(img, dim)
        
        # Create a CLAHE object (contrast limiting adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=5.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img.astype("uint8"))

    
        
        if vo.update(img, img_id) and img_id < 1118 :  # Updating the vectors in VisualOdometry class
            
            print(img_id)
            
            if img_id == 0:
                T_v_dict[img_path] = ([[0], [0], [0]])
            else:
                T_v_dict[img_path] = vo.cur_t   # Retrieve the translation vectors for dictionary
                count += 1
            cur_t = vo.cur_t  # Retrieve the translation vectors

            # ------- Windowed Displays ---------
            # if window_flag == 'WINDOW_YES':
            #     if img_id > 0 and img_id <=1119:  # Set the points for the real-time trajectory window
            #         x, y, z = cur_t[0], cur_t[1], cur_t[2]
            #         #TT.drawOpticalFlowField(imgKLT, vo.OFF_prev, vo.OFF_cur)  # Draw the features that were matched
            #     else:
            #         x, y, z = 0., 0., 0.

                #traj = TT.RT_trajectory_window(traj, x, y, z, img_id)  # Draw the trajectory window
            # -------------------------------------
           
            img_id += 1  # Increasing the image id
            #print("elmoshkela hena?")
            

    # Write poses to text file in the images sequences directory
    poses_MVO = open("/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/py-MVO_Poses.txt", 'w')
    poses_MVO.truncate(0)
    for t_v, R_m in zip(vo.T_vectors, vo.R_matrices):
        T = np.hstack((R_m, t_v)).flatten()
        for t in T:
           poses_MVO.write(' '.join([str(t) for t in T]) + '\n')
            
    #poses_MVO.close()  # Close the Poses text file

    # Write the images path and translation vector to text file in the images sequences directory
    VO_t = open("/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/images/py-MVO_poses.txt", 'w')
    # Retrieving the translation vectors from the
    # translation vector dictionary and write it in a txt file
    T_v = []
    for key, value in T_v_dict.items():
        T_v_dict[key] = np.array((value[0][0], value[2][0]))
        T_v.append((value[0][0], value[2][0]))
        VO_t.write( str(value[0][0]) + ' ' + str(value[2][0])+ ' ')
    #print(T_v[0], T_v[1], T_v[2])
    
    #this writes 50 arrays to the csv file
    # letsgetthisshit = np.loadtxt("/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/images/py-MVO_poses.txt")   
    # resultarray = []
    # counter = 0
    # topofthemorning = np.zeros(12, dtype=object)

    # for i in range(0, 600):
    #     topofthemorning[counter] = letsgetthisshit[i]
    #     counter +=1
    #     if counter ==12:
    #         resultarray.append(topofthemorning)
    #         topofthemorning = np.zeros(12)
    #         counter = 0
                
    # np.savetxt('/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikedata/bikedata.csv', resultarray)
    # np.savetxt('/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikedata/bikedata.csv', resultarray)

    # loading 50 arrays from gt
    #wholegt = np.loadtxt('/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/mygeodata/map.txt', skiprows=0, max_rows=50)
    
    #np.savetxt('/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/bikedata/bikedata.csv', wholegt)


     

    #VO_t.close()  # Close the Poses text file

    # -------- Plotting Trajectories ----------
    if window_flag == 'WINDOW_YES' or window_flag == 'WINDOW_T':

        # if GT_poses:  # Ground Truth Data is used in case GPS and GT are available
        #     # Ground Truth poses in list
        #     GT_poses = GT.ground_truth(GT_poses)
        #     # Plot VO and ground truth trajectories
        #     TT.VO_GT_plot(T_v, GT_poses)

        #elif gps_switch:  # Plotting the VO and GPS trajectories
            #if GPS_flag == 'GPS_T':
                #TT.GPS_VO_plot(T_v, utm_dict)
            #elif GPS_flag == 'GPS_T_M':
                # Do merged trajectory
                #VO_dict = TT.GPS_VO_Merge_plot(T_v_dict, utm_dict)

                # Write GPS to text file in the images sequences directory
                #VO_utm_coord = open("/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/py-MVO_Poses.txt", 'w')
                #for key, value in VO_dict.items():
                    #value = to_latlon(value[0], value[1], 17, 'U')
                    #VO_utm_coord.write(key+' '+str(value[0])+' '+str(value[1])+'\n')
                #VO_utm_coord.close()  # Close the Poses text file

        
            TT.VO_plot(T_v)
            
            
 
   
    return



if __name__ == '__main__':

    run()


