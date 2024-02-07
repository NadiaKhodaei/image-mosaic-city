import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import random
from tqdm.notebook import tqdm


class Panaroma:

    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=0.9,match_status=False):

        #detect the features and keypoints from SIFT
        (imageB, imageA) = images
        (KeypointsA, features_of_A) = self.Detect_Feature_And_KeyPoints(imageA)
        (KeypointsB, features_of_B) = self.Detect_Feature_And_KeyPoints(imageB)

        #got the valid matched points
        Values = self.matchKeypoints(KeypointsA, KeypointsB,features_of_A, features_of_B, lowe_ratio, max_Threshold)

        if Values is None:
            return None

        #to get perspective of image using computed homography
        (matches, Homography, status) = Values
        result_image = self.getwarp_perspective(imageA,imageB,Homography)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        blender = Blender()
        
        
        result_image = blender.linearBlending([imageB, result_image])
        result_image = self.removeBlackBorder(result_image)
        # result_image = result_image[y:y+h, x:x+w]
        # check to see if the keypoint matches should be visualized
        if match_status:
            vis = self.draw_Matches(imageA, imageB, KeypointsA, KeypointsB, matches,status)
            return (result_image, vis)
  
        return result_image

    def getwarp_perspective(self,imageA,imageB,Homography):
        val = imageA.shape[1] + imageB.shape[1]
        result_image = cv2.warpPerspective(imageA, Homography, (val , imageA.shape[0]))
    
        return result_image

    def Detect_Feature_And_KeyPoints(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect and extract features from the image
        descriptors = cv2.xfeatures2d.SIFT_create()
        (Keypoints, features) = descriptors.detectAndCompute(image, None)

        Keypoints = np.float32([i.pt for i in Keypoints])
        return (Keypoints, features)

    def get_Allpossible_Match(self,featuresA,featuresB):

        # compute the all matches using euclidean distance and opencv provide
        #DescriptorMatcher_create() function for that
        match_instance = cv2.DescriptorMatcher_create("BruteForce")
        All_Matches = match_instance.knnMatch(featuresA, featuresB, 2)

        return All_Matches

    def All_validmatches(self,AllMatches,lowe_ratio):
        #to get all valid matches according to lowe concept..
        valid_matches = []

        for val in AllMatches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))

        return valid_matches
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
        
        return img[:reduced_h, :reduced_w]
    def Compute_Homography(self,pointsA,pointsB,max_Threshold):
        #to compute homography using points in both images

        (H, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return (H,status)

    def matchKeypoints(self, KeypointsA, KeypointsB, featuresA, featuresB,lowe_ratio, max_Threshold):

        AllMatches = self.get_Allpossible_Match(featuresA,featuresB)
        valid_matches = self.All_validmatches(AllMatches,lowe_ratio)

        if len(valid_matches) > 4:
            # construct the two sets of points
            pointsA = np.float32([KeypointsA[i] for (_,i) in valid_matches])
            pointsB = np.float32([KeypointsB[i] for (i,_) in valid_matches])

            (Homograpgy, status) = self.Compute_Homography(pointsA, pointsB, max_Threshold)

            return (valid_matches, Homograpgy, status)
        else:
            return None

    def get_image_dimension(self,image):
        (h,w) = image.shape[:2]
        return (h,w)

    def get_points(self,imageA,imageB):

        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        return vis

    
    def draw_Matches(self, imageA, imageB, KeypointsA, KeypointsB, matches, status):

        (hA,wA) = self.get_image_dimension(imageA)
        vis = self.get_points(imageA,imageB)

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(KeypointsA[queryIdx][0]), int(KeypointsA[queryIdx][1]))
                ptB = (int(KeypointsB[trainIdx][0]) + wA, int(KeypointsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

    
    def SIFT(self, image):
        siftDetector= cv2.xfeatures2d.SIFT_create() # limit 1000 points
        # siftDetector= cv2.SIFT_create()  # depends on OpenCV version

        kp, des = siftDetector.detectAndCompute(image, None)
        return kp, des
    def plot_sift(self, image, kp):
        tmp = image.copy()
        img = cv2.drawKeypoints(image, kp, tmp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return img

    def matcher(self, kpp1, dess1, img1, kpp2, dess2, img2, threshold):
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(dess1,dess2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < threshold*n.distance:
                good.append([m])

        matches = []
        for pair in good:
            matches.append(list(kpp1[pair[0].queryIdx].pt + kpp2[pair[0].trainIdx].pt))

        matches = np.array(matches)
        return matches


    def plot_matches(self, matches, total_img):
        match_img = total_img.copy()
        offset = total_img.shape[1]/2
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.imshow(np.array(match_img).astype('int')) #　RGB is integer type
        
        ax.plot(matches[:, 0], matches[:, 1], 'xr')
        ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
        
        ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
                'r', linewidth=0.6)

        # plt.show()
    

    def homography(self, pairs):
        rows = []
        for i in range(pairs.shape[0]):
            p1 = np.append(pairs[i][0:2], 1)
            p2 = np.append(pairs[i][2:4], 1)
            row1 = [0, 0, 0, p1[0], p1[1], p1[2], -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*p1[2]]
            row2 = [p1[0], p1[1], p1[2], 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*p1[2]]
            rows.append(row1)
            rows.append(row2)
        rows = np.array(rows)
        U, s, V = np.linalg.svd(rows)
        H = V[-1].reshape(3, 3)
        H = H/H[2, 2] # standardize to let w*H[2,2] = 1
        return H
    

    def get_error(self, points, H):
        num_points = len(points)
        all_p1 = np.concatenate((points[:, 0:2], np.ones((num_points, 1))), axis=1)
        all_p2 = points[:, 2:4]
        estimate_p2 = np.zeros((num_points, 2))
        for i in range(num_points):
            temp = np.dot(H, all_p1[i])
            estimate_p2[i] = (temp/temp[2])[0:2] # set index 2 to 1 and slice the index 0, 1
        # Compute error
        errors = np.linalg.norm(all_p2 - estimate_p2 , axis=1) ** 2

        return errors
    def random_point(self, matches, k=4):
        idx = random.sample(range(len(matches)), k)
        point = [matches[i] for i in idx ]
        return np.array(point)


    def ransac(self, matches, threshold, iters):
            num_best_inliers = 0
            
            for i in range(iters):
                points = self.random_point(matches)
                H = self.homography(points)
                
                #  avoid dividing by zero 
                if np.linalg.matrix_rank(H) < 3:
                    continue
                    
                errors = self.get_error(matches, H)
                idx = np.where(errors < threshold)[0]
                inliers = matches[idx]

                num_inliers = len(inliers)
                if num_inliers > num_best_inliers:
                    best_inliers = inliers.copy()
                    num_best_inliers = num_inliers
                    best_H = H.copy()
                    
            print("inliers/matches: {}/{}".format(num_best_inliers, len(matches)))
            return best_inliers, best_H
 
      
                
class Blender:
    def linearBlending(self, images):
        '''
        linear Blending(also known as Feathering)
        '''
        img_left, img_right = images
        (hl, wl) = img_left.shape[:2]
        (hr, wr) = img_right.shape[:2]
        img_left_mask = np.zeros((hr, wr), dtype="int")
        img_right_mask = np.zeros((hr, wr), dtype="int")
        
        # find the left image and right image mask region(Those not zero pixels)
        for i in range(hl):
            for j in range(wl):
                if np.count_nonzero(img_left[i, j]) > 0:
                    img_left_mask[i, j] = 1
        for i in range(hr):
            for j in range(wr):
                if np.count_nonzero(img_right[i, j]) > 0:
                    img_right_mask[i, j] = 1
        
        # find the overlap mask(overlap region of two image)
        overlap_mask = np.zeros((hr, wr), dtype="int")
        for i in range(hr):
            for j in range(wr):
                if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                    overlap_mask[i, j] = 1
        
        # Plot the overlap mask
        # plt.figure(21)
        # plt.title("overlap_mask")
        # plt.imshow(overlap_mask.astype(int), cmap="gray")
        
        # compute the alpha mask to linear blending the overlap region
        alpha_mask = np.zeros((hr, wr)) # alpha value depend on left image
        for i in range(hr): 
            minIdx = maxIdx = -1
            for j in range(wr):
                if (overlap_mask[i, j] == 1 and minIdx == -1):
                    minIdx = j
                if (overlap_mask[i, j] == 1):
                    maxIdx = j
            
            if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                continue
                
            decrease_step = 1 / (maxIdx - minIdx)
            for j in range(minIdx, maxIdx + 1):
                alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
        
        
        
        linearBlending_img = np.copy(img_right)
        linearBlending_img[:hl, :wl] = np.copy(img_left)
        # linear blending
        for i in range(hr):
            for j in range(wr):
                if ( np.count_nonzero(overlap_mask[i, j]) > 0):
                    linearBlending_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
        
        return linearBlending_img





