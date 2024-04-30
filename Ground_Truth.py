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

"""

from itertools import filterfalse
import math
from tkinter.font import BOLD

from utm import from_latlon

import Trajectory_Tools as TT
from Common_Modules import *

# def ground_truth(folder):
#     """ Obtain the Ground Truth poses can also be used with the
#      py-MVO_Poses text file"""
#     with open(folder, "r") as f:
#         txt_file = f.readlines()

#     l_poses = []
#     for num, i in enumerate(txt_file):
#         i = i.strip().split()

#         x = float(i[3])
#         y = float(i[7])
#         z = float(i[11])

#         l_poses.append([x, z])

#     return l_poses


''''def plot_ground_truth(l_poses):
    """ Plots the ground truth or py-MVO_Poses data"""
    font = {'family': 'DejaVu Sans',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)
    plt.figure(1)
    plt.plot(*zip(*l_poses), color='red', marker='o')

    # Set plot parameters and show it
    plt.axis('equal')
    plt.grid()
    plt.show()

def plot_2_ground_truth(pose1, pose2):
    """ Plots both, the ground truth and py-MVO_Poses data"""
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)
    fig = plt.figure(1)
    plt.plot(*zip(*pose1), color='red', marker='o')
    plt.plot(*zip(*pose2), color='blue', marker='o')
    # Set plot parameters and show it
    plt.axis('equal')
    plt.grid()
    plt.show()




def plot_GPS_and_TV(GPS_file, VO_poses):
    """ Plots the raw_GPS/py-MVO_GPS and the py-MVO_TV text files """

    vo_poses = OrderedDict()
    # Reading in the textfile
    with open(VO_poses, "r") as textfile:
        for line in textfile:
            parse_line = line.strip().split()
            vo_poses[parse_line[0]] = (float(parse_line[1]), float(parse_line[2]))

    coordinates_dict = OrderedDict()
    # Reading in the textfile
    with open(GPS_file, "r") as textfile:
        for line in textfile:
            parse_line = line.strip().split()
            coordinates_dict[parse_line[0]] = from_latlon(float(parse_line[1]), float(parse_line[2]), 17)[:2]

    k = vo_poses.keys() + coordinates_dict.keys()
    k = [i for i in k if k.count(i) > 1]

    T_v, gps_orig = [], []
    for key in k:
        T_v.append(vo_poses[key])
        gps_orig.append(coordinates_dict[key])

    # Retrieving the GPS coordinates into a list
    # Shifting the trajectory to the origin
    utm_dx = gps_orig[0][0]
    utm_dy = gps_orig[0][1]

    gps = [(u[0] - utm_dx, u[1] - utm_dy) for u in gps_orig]

    last_gps = gps[len(gps) - 1]
    last_vo = T_v[len(T_v) - 1]
    d_gps = math.sqrt((last_gps[0] ** 2) + (last_gps[1] ** 2))
    d_VO = math.sqrt((last_vo[0] ** 2) + (last_vo[1] ** 2))


    scale = d_gps / d_VO

    # print 'The scale factor', scale
    # Apply scale factor to the translation vectors
    T_v = [np.array(t) * scale for t in T_v]

    # Obtaining the angle between the first points of each list: VO list and GPS list
    rotate_idx = min(len(T_v) - 1, len(gps) - 1)
    VO_v = np.array(T_v[rotate_idx])
    GPS_v = np.array(gps[rotate_idx])

    # Distance between points.
    d1 = math.sqrt((VO_v[0] - GPS_v[0]) ** 2 + (VO_v[1] - GPS_v[1]) ** 2)
    # Obtain the angle assuming the two points are vectors
    angle = math.acos((VO_v.dot(GPS_v)) / (np.linalg.norm(VO_v) * np.linalg.norm(GPS_v)))
    # Rotates the GPS point only for verification
    VO_v = TT.rotateFunct([VO_v], angle)

    # Distance between points after rotation.
    d2 = math.sqrt((VO_v[0][0] - GPS_v[0]) ** 2 + (VO_v[0][1] - GPS_v[1]) ** 2)
    # Verify if points are closer after rotation if not rotate the other way
    if d2 < d1:
        sign = 1
    else:
        sign = -1

    # Rotating the GPS function so it aligns with the VO function
    T_v = TT.rotateFunct(T_v, sign * angle)
    
    print(len(T_v))
    print(len(gps_orig))

    # --------------------------------------------------

    # Plotting the VO and GPS trajectories
    

    
    plt.rc('font', **font)[1]+'\n', 'weight', 'size'
    plt.figure(1)
    GPS, = plt.plot(*zip(*gps), color='red', marker='o', label='GPS')
    pyMVO, = plt.plot(*zip(*T_v)), 'weight', 'size' 
    plt.legend(handles=[pyMVO, GPS])
    # Set plot parameters and show it
    plt.axis('equal')
    plt.grid()
    font = {'family': 'normal', 'weight': 'bold', 'size': '22'}
    plt.show()
  '''   
def run():
        """ Place run() function in main on 'bold'this file in
        order to plot the Ground Truth data only. """

        # vo_txt = "/home/hana/Desktop/Bachelors/repos/1/visual_odometry-master/src/KITTI_sample/mygeodata/map.txt"
        # vo_poses = ground_truth(vo_txt)
        
        #plot_ground_truth(vo_poses)
                #plot_GPS_and_TV(gps_txt, vo_txt)
        #plot_2_ground_truth(vo_poses, vo_txt)
        #print(ground_truth(vo_poses))

        #else:
        #plot_ground_truth(vo_poses)


        if __name__ == '__main__':
                run();
