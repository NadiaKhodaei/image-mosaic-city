from function import Panaroma
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from stitching import AffineStitcher
from stitching import AffineStitcher
from stitching import Stitcher

print("Enter the number of images you want to use:")
no_of_images = int(input())
print("Enter the image name")

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


filename = []

for i in range(no_of_images):
    print("Enter the %d image:" %(i+1))
    filename.append(input())

images = []

for i in range(no_of_images):
    images.append(cv2.imread(filename[i]))



for i in range(no_of_images):
    images[i] = imutils.resize(images[i], width=1000)

for i in range(no_of_images):
    images[i] = imutils.resize(images[i], height=1000)


panaroma = Panaroma()
if no_of_images==2:
    (result, matched_points) = panaroma.image_stitch([images[0], images[1]], match_status=True)

else:
    (result, matched_points) = panaroma.image_stitch([images[no_of_images-2], images[no_of_images-1]], match_status=True)
    for i in range(no_of_images - 2):
        (result, matched_points) = panaroma.image_stitch([images[no_of_images-i-3],result], match_status=True)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()

imag1=images[0]
imag2=images[1]
if no_of_images == 2:
    imag3=images[1]
else:
    imag3=images[no_of_images-1]

kp1, des1 = panaroma.SIFT(imag1)
kp2, des2 = panaroma.SIFT(imag2)
kp3, des3 = panaroma.SIFT(imag3)


kp1_img = panaroma.plot_sift(imag1, kp1)

kp2_img = panaroma.plot_sift(imag2, kp2)
kp3_img = panaroma.plot_sift(imag3, kp3)

total_kp = np.concatenate((kp1_img,kp2_img), axis=1)

# plot_image(total_kp, (20,20))

matches = panaroma.matcher(kp1, des1, imag1, kp2, des2, imag2, 0.5)

total_img2 = np.concatenate((kp1_img,kp2_img), axis=1)
panaroma.plot_matches(matches, total_img2) # Good mathces
match_img = total_img2.copy()
offset = total_img2.shape[1]/2

inliers, H = panaroma.ransac(matches, 0.5, 500)
# panaroma.plot_matches(inliers, total_img2)



stitcher = Stitcher()
settings = {    "crop": True,
                "matcher_type": 'homography',
                "finder":'dp_color',
                "blender_type": 'multiband',
                "confidence_threshold": 0.9}    

final = AffineStitcher(**settings)

final_result = final.stitch(images[0:no_of_images])
# plot_image(final_result, (20,20))


rows = 2 
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(10,10) , squeeze=False)
fig.suptitle('automatic mosaic')
i = 0



for r in range(rows):
    for c in range(cols): 
        axs[0, 0].set_aspect('auto')
        axs[0, 0].imshow(total_kp)
        axs[0, 0].set_title('image1 keypoints')

        axs[1, 0].set_aspect('auto')
        axs[1, 0].imshow(np.array(total_img2).astype('int')) 
        
        axs[1, 0].plot(matches[:, 0], matches[:, 1], 'xr')
        axs[1, 0].plot(matches[:, 2] + offset, matches[:, 3], 'xr')
        
        axs[1, 0].plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
                'r', linewidth=0.8)
        axs[1, 0].set_title('deature matching without Ransac')
        axs[1, 1].set_aspect('auto')
        axs[1, 1].imshow(np.array(total_img2).astype('int')) 
        axs[1, 1].plot(inliers[:, 0], inliers[:, 1], 'xr')
        axs[1, 1].plot(inliers[:, 2] + offset, inliers[:, 3], 'xr')
        
        axs[1, 1].plot([inliers[:, 0], inliers[:, 2] + offset], [inliers[:, 1], inliers[:, 3]],
                'r', linewidth=0.8)
        axs[1, 1].set_title('deature matching with Ransac')
        axs[0, 1].set_aspect('auto')
        axs[0, 1].imshow(final_result)
        axs[0, 1].set_title('Result')


        # axs[0, 3].panaroma.plot_matches(matches, total_img2)
        # axs[0, 3].set_title('Result')
        i = i + 1     
plt.show()
plt.close()
