import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
cbrow = 4
cbcol = 11

objp = np.zeros((cbrow*cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Images/Microsoft/PolkaDot/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    print(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findCirclesGrid(gray, (cbcol,cbrow),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawCirclesGrid(img, (cbcol,cbrow), corners2,ret)
        cv2.imshow('img', img)
        cv2.imwrite(fname + '-processed.jpg', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(fname+'calibresult.png',dst)
print ret
print "\n\n"
print mtx 
print "\n\n"
print dist 
print "\n\n"
print rvecs 
print "\n\n"
print tvecs 

# cv2.destroyAllWindows()