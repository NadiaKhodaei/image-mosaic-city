from stitching import Stitcher
from stitching.images import Images
from pathlib import Path
from matplotlib import pyplot as plt
import cv2 as cv
from stitching import AffineStitcher
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher


def plot_image(img, figsize_in_inches=(100,100)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
# def plot_images(imgs, figsize_in_inches=(5,5)):
#     fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
#     for col, img in enumerate(imgs):
#         axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#     plt.show()

def get_image_paths(img_set):
    return [str(path.relative_to('.')) for path in Path('imgs').rglob(f'{img_set}*')]

  
weir_imgs = get_image_paths('weir')

images = Images.of(weir_imgs)

medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))

finder = FeatureDetector(detector='orb', nfeatures=500)
features = [finder.detect_features(img) for img in medium_imgs]

keypoints_center_img = finder.draw_keypoints(medium_imgs[1], features[1])
# plot_image(keypoints_center_img, (20,10))


matcher = FeatureMatcher(matcher_type='homography', range_width=-1)
matches = matcher.match_features(features)
all_relevant_matches = matcher.draw_matches_matrix(medium_imgs, features, matches, conf_thresh=1, 
                                                   inliers=True, matchColor=(0, 255, 0))





    
stitcher = Stitcher()
settings = {# The whole plan should be considered
            "crop": True,
            # The matches confidences aren't that good
            "confidence_threshold": 0.5}    

stitcher = AffineStitcher(**settings)
panorama = stitcher.stitch(weir_imgs)

rows = 3 
cols = 1
fig, axs = plt.subplots(rows, cols, figsize=(20,20) , squeeze=False)
fig.suptitle('automatic mosaic')
i = 0
for r in range(rows):
    for c in range(cols): 
        axs[0, 0].imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
        axs[0, 0].set_title('panorma')
        axs[1, 0].imshow(cv.cvtColor(keypoints_center_img, cv.COLOR_BGR2RGB))
        axs[1, 0].set_title('keypoints feature')
        for idx1, idx2, img in all_relevant_matches:
            print(f"Matches Image {idx1+1} to Image {idx2+1}")
            axs[2, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            axs[2, 0].set_title('keypoints feature')
        i = i + 1     
plt.show()
plt.close()

