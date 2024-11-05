import cv2
import glob
import numpy as np
import math

# Step 1: Camera Calibration

# Prepare object points
objp = np.zeros((7 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
obj_points = []  # 3D points in real-world space
img_points = []  # 2D points in image plane

# Find checkerboard corners in each image
image_folder = "./HD_input_frames/*.jpg"  # Update the folder and file extension if needed
images = glob.glob(image_folder)
for img_path in images:
    img = cv2.imread(img_path)
    
    # Preprocess the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find corners with subpixel accuracy
    ret, corners = cv2.findChessboardCorners(thresh, (7, 7), None)
    
    if ret:
        obj_points.append(objp)
        
        # Refine corner locations
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners_refined)

# Calibrate the camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the intrinsic matrix
print("Intrinsic Matrix (K):")
print(K)
print("Distortion Coefficients:")
print(dist)

# Step 2: Distance Estimation

# Select the desired frame
frame_path = "./HD_input_frames/hdframe_5.jpg"  # Update with the path to the desired frame

# Load the frame
frame = cv2.imread(frame_path)
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the frame for point selection
cv2.imshow('Select Points', frame)

# Create variables to store the selected points
P1 = None
P2 = None

# Mouse click callback function
def mouse_callback(event, x, y, flags, param):
    global P1, P2

    if event == cv2.EVENT_LBUTTONDOWN:
        if P1 is None:
            P1 = np.array([x, y, 1])
            print("Point 1: ", P1)
        elif P2 is None:
            P2 = np.array([x, y, 1])
            print("Point 2: ", P2)

# Set the mouse callback function
cv2.setMouseCallback('Select Points', mouse_callback)

# Wait for the user to select the points
cv2.waitKey(0)

# Check if points are selected
if P1 is None or P2 is None:
    print("Points not selected. Moving to the next frame.")
    cv2.destroyAllWindows()

# Display and save the image with marked points
img_with_points = frame.copy()
cv2.circle(img_with_points, tuple(P1[:2]), 5, (0, 0, 255), -1)  # Mark P1 with a red circle
cv2.circle(img_with_points, tuple(P2[:2]), 5, (0, 255, 0), -1)  # Mark P2 with a green circle
cv2.putText(img_with_points, "P1", tuple(P1[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Label P1
cv2.putText(img_with_points, "P2", tuple(P2[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label P2
cv2.imshow('Image with Points', img_with_points)
cv2.imwrite('image_with_points.jpg', img_with_points)
cv2.waitKey(0)

# Calculate the inverse of the intrinsic matrix
intrinsic_inv = np.linalg.inv(K)
print("Inverse matrix:", intrinsic_inv)

# Calculate A1 and A2
A1 = np.matmul(intrinsic_inv, P1)
A2 = np.matmul(intrinsic_inv, P2)
print("A1:", A1)
print("A2:", A2)

# Calculate C1 and C2
C1 = np.linalg.norm(A1)
C2 = np.linalg.norm(A2)
print("C1:", C1)
print("C2:", C2)

# Calculate the angle alpha
alpha = np.arccos(np.dot(A1, A2) / (C1 * C2))
print("Alpha:", alpha)

# Width of checkerboard pattern in meters
checkerboard_width = 0.506

# Calculate the distance X
X = (checkerboard_width / 2) / np.tan(alpha / 2)
print("Distance:", X)

# Close all OpenCV windows
cv2.destroyAllWindows()