import cv2
import numpy as np

# Read the reference object image and the scene image
reference_object = cv2.imread('vitaminc.jpg', 0)  # Read as grayscale
scene_image = cv2.imread('img_query_2.jpg')
print(reference_object)
print(scene_image)
# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(reference_object, None)
keypoints2, descriptors2 = orb.detectAndCompute(scene_image, None)

# Create a Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

# Find Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Define the corners of the reference object
h, w = reference_object.shape
object_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)

# Apply perspective transformation to the reference object corners
transformed_corners = cv2.perspectiveTransform(object_corners, M)

# Draw a bounding box around the detected object
scene_image_with_boxes = scene_image.copy()

for corner in transformed_corners:
    x, y = corner[0]
    cv2.circle(scene_image_with_boxes, (int(x), int(y)), 5, (0, 255, 0), -1)

# Define names for objects
object_names = ['airpods 1']  # Replace with actual object names

# Display labels with bounding boxes
for i, corner in enumerate(transformed_corners):
    x, y = corner[0]
    object_name = object_names[i] if i < len(object_names) else f'Object {i + 1}'
    cv2.putText(scene_image_with_boxes, object_name, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(scene_image_with_boxes, f'({int(x)}, {int(y)})', (int(x), int(y) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show the detected objects
cv2.imshow('Detected Objects', scene_image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
