import numpy as np
import cv2
import matplotlib.pyplot as plt


def removeBlackBorder(self, img):
        '''
        Remove img's the black border 
        '''
        h, w = img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
# Feature Detection and Description
def detect_and_describe(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Feature Matching
def match_keypoints(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=60)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Match des1 and des2
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.83 * n.distance]
    
    return good_matches

# Homography Estimation
def estimate_homography(kp1, kp2, des1, des2):
    # Match keypoints and descriptors between des1 and des2
    good_matches = match_keypoints(des1, des2)
    
    if not good_matches:
        print("No valid matches found between the images.")
        return None

    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Use RANSAC to find the homography matrix
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Image Warping and Blending
def warp_and_blend_images(img1, img2, homography):
    # Warp the second image to align with the first image
    result = cv2.warpPerspective(img1, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    return result

# Read images
img3 = cv2.imread('11.jpg')
img2 = cv2.imread('12.jpg')
img1 = cv2.imread('13.jpg')

# Feature Detection and Description
kp1, des1 = detect_and_describe(img1)
kp2, des2 = detect_and_describe(img2)
kp3, des3 = detect_and_describe(img3)

# Feature Matching and Blending for Image 1 and Image 2
good_matches1_2 = match_keypoints(des1, des2)
homography1_2 = estimate_homography(kp1, kp2, des1, des2)
result1_2 = warp_and_blend_images(img1, img2, homography1_2)

# Feature Matching and Blending for Result of Image 1 and 2 and Image 3
kp_result, des_result = detect_and_describe(result1_2)
good_matches_result_3 = match_keypoints(des_result, des3)
homography_result_3 = estimate_homography(kp_result, kp3, des_result, des3)
final_result = warp_and_blend_images(result1_2, img3, homography_result_3)
final_result = removeBlackBorder(final_result)
# Display the final result
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.show()

'''with plot '''

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Feature Detection and Description
def detect_and_describe(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Feature Matching
def match_keypoints(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
    return good_matches

# Homography Estimation
def estimate_homography(kp1, kp2, good_matches):
    src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return homography

# Image Warping and Blending with Trim
def warp_and_blend_images(img3, img2, homography):
    result = cv2.warpPerspective(img3, homography, (img3.shape[1] + img2.shape[1], img3.shape[0]))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    # Trim zero pixels from the border of the result
    non_zero_rows = np.any(result, axis=1)
    non_zero_cols = np.any(result, axis=0)

    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]

    result_trimmed = result[row_start:row_end + 1, col_start:col_end + 1]

    return result_trimmed

# Read images
img3 = cv2.imread('image3.jpg')
img2 = cv2.imread('image4.jpg')
img1 = cv2.imread('image5.jpg')

# Feature Detection and Description
kp1, des1 = detect_and_describe(img1)
kp2, des2 = detect_and_describe(img2)
kp3, des3 = detect_and_describe(img3)

# Feature Matching and Blending for Image 1 and Image 2
good_matches1_2 = match_keypoints(des1, des2)
homography1_2 = estimate_homography(kp1, kp2, good_matches1_2)
result1_2 = warp_and_blend_images(img1, img2, homography1_2)

# Feature Matching and Blending for Result of Image 1 and 2 and Image 3
kp_result, des_result = detect_and_describe(result1_2)
good_matches_result_3 = match_keypoints(des_result, des3)
homography_result_3 = estimate_homography(kp_result, kp3, good_matches_result_3)
final_result = warp_and_blend_images(result1_2, img3, homography_result_3)
final
# Plotting in Subplots
plt.figure(figsize=(15, 10))

# Plot Keypoints for Image 1
plt.subplot(2, 3, 1)
img1_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(img1_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Keypoints for Image 1')

# Plot Keypoints for Image 2
plt.subplot(2, 3, 2)
img2_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(img2_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Keypoints for Image 2')

# Plot Good Matches for Image 1 and Image 2
plt.subplot(2, 3, 3)
img_matches1_2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, [good_matches1_2], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 0, 255))
plt.imshow(cv2.cvtColor(img_matches1_2, cv2.COLOR_BGR2RGB))
plt.title('Good Matches between Image 1 and Image 2')

# Plot Result of Image 1 and Image 2 Blending
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(result1_2, cv2.COLOR_BGR2RGB))
plt.title('Result of Image 1 and Image 2 Blending')

# Plot Keypoints for Result of Image 1 and 2
plt.subplot(2, 3, 5)
result_keypoints = cv2.drawKeypoints(result1_2, kp_result, None, color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(result_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Keypoints for Result of Image 1 and Image 2')

# Plot Good Matches for Result of Image 1 and 2 and Image 3
plt.subplot(2, 3, 6)
img_matches_result_3 = cv2.drawMatchesKnn(result1_2, kp_result, img3, kp3, [good_matches_result_3], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(255, 0, 255))
plt.imshow(cv2.cvtColor(img_matches_result_3, cv2.COLOR_BGR2RGB))
plt.title('Good Matches between Result of Image 1 and Image 2 and Image 3')

plt.tight_layout()
plt.show()
