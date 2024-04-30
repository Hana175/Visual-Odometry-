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


from Common_Modules import *
import datetime
import py_MVO as OF
import imutils
from imutils.video import VideoStream
from cv2 import ORB, SIFT, BFMatcher, SIFT_create


# CONSTANTS
fMATCHING_DIFF = 1  # Minimum difference in the KLT point correspondence

lk_params = dict(winSize=(21, 21),  # Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
                 maxLevel=2, 
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def KLT_featureTracking(image_ref, image_cur, px_ref):
    """Feature tracking using the Kanade-Lucas-Tomasi tracker.
    A backtracking check is done to ensure good features. The backtracking features method
    consist of tracking a set of features, f-1, onto a new frame, which will produce the corresponding features, f-2,
    on the new frame. Once this is done we take the f-2 features, and look for their
    corresponding features, f-1', on the last frame. When we obtain the f-1' features we look for the
    absolute difference between f-1 and f-1', abs(f-1 and f-1'). If the absolute difference is less than a certain
    threshold(in this case 1) then we consider them good features."""

    # Feature Correspondence with Backtracking Check
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params,minEigThreshold = 1e-4)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params,minEigThreshold = 1e-4)

    d = abs(px_ref - kp1).reshape(-1, 2).max(-1)  # Verify the absolute difference between feature points
    good = d < fMATCHING_DIFF  # Verify which features produced good results by the difference being less
                               # than the fMATCHING_DIFF threshold.
    diff_mean = np.mean(d)
    # Error Management
    if len(d) == 0:
        print('Error: No matches where made.')
    elif list(good).count(True) <= 3:  # If less than 5 good points, it uses the features obtain without the backtracking check
        print('Warning: No match was good. Returns the list without good point correspondence.')
        return kp1, kp2, diff_mean

    # Create new lists with the good features
    n_kp1, n_kp2 = [], []
    for i, good_flag in enumerate(good):
        if good_flag:
            n_kp1.append(kp1[i])
            n_kp2.append(kp2[i])

    # Format the features into float32 numpy arrays
    n_kp1, n_kp2 = np.array(n_kp1, dtype=np.float32), np.array(n_kp2, dtype=np.float32)

    # Verify if the point correspondence points are in the same
    # pixel coordinates. If true the car is stopped (theoretically)
    d = abs(n_kp1 - n_kp2).reshape(-1, 2).max(-1)

    # The mean of the differences is used to determine the amount
    # of distance between the pixels
    

    return n_kp1, n_kp2, diff_mean


def betterMatches(F, points1, points2):
    """ Minimize the geometric error between corresponding image coordinates.
    For more information look into OpenCV's docs for the cv2.correctMatches function."""
    if OF.self.F_detectors == 'ORB': #trying brute force for ORB
                print("im here")
                kp1, des1 = ORB.detectAndCompute(prev_img,None)
                kp2, des2 = ORB.detectAndCompute(cur_img,None)
                brute_force = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
                matches = brute_force.match(des1,des2)
                matches = sorted(matches, key = lambda x:x.distance)
                print("im here 2")
                # finding the humming distance of the matches and sorting them
                img3 = cv2.drawMatches(OF.prev_img,kp1,OF.cur_img,kp2,matches[:10], flags=2)
                print("almost done?")
                plt.imshow(img3),plt.show()
                print("done??")
    if OF.self.F_detectors == 'SIFT':
            kp1, des1 = SIFT.detectAndCompute(prev_img,None)
            kp2, des2 = SIFT.detectAndCompute(cur_img,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2, k=2) #knn is k best matches
            good = [] #this is a ratio test if its a good match i add it to the good array
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            img3 = cv2.drawMatchesKnn(OF.prev_img,kp1,OF.cur_img,kp2,good,flags=2)
            plt.imshow(img3),plt.show()
    # Reshaping for cv2.correctMatches
    kp1 = np.reshape(points1, (1, points1.shape[0], 2))
    kp2 = np.reshape(points2, (1, points2.shape[0], 2))

    newPoints1, newPoints2 = cv2.correctMatches(F, points1, points2)

    return newPoints1[0], newPoints2[0]





