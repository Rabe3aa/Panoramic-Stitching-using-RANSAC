import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib

def reading_image(path):    
    img = cv2.imread(path)
    return img

def gray_image(img):
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cv2.COLOR_BGR2GRAY)
    return img

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15  
zoom_x1 = 300; zoom_x2 = 1300
zoom_y1 = 300; zoom_y2 = 700
im1Keypoints = np.array([])
im2Keypoints = np.array([])

image1 = reading_image("/content/Picture1.jpg")
image2 = reading_image("/content/Picture2.jpg")

image_gray_1 = gray_image(image1)
image_gray_2 = gray_image(image2)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)

keypoints1, descriptors1 = orb.detectAndCompute(image_gray_1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image_gray_2, None)

im1Keypoints = cv2.drawKeypoints(image1, keypoints1, im1Keypoints, color=(0,0,255),flags=0)
im2Keypoints = cv2.drawKeypoints(image2, keypoints2, im2Keypoints, color=(0,0,255),flags=0)

cv2.imwrite("keypoints1.jpg", im1Keypoints)
cv2.imwrite("keypoints2.jpg", im2Keypoints)

fig = plt.figure(figsize=(15, 15))
fig.add_subplot(1, 2, 1)

plt.imshow(im1Keypoints[:,:,::-1])
plt.title("image1 key points")

fig.add_subplot(1, 2, 2)

plt.imshow(im2Keypoints[:,:,::-1])
plt.title("image2 key points")

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)

# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]

# Draw top matches
imMatches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)
cv2.imwrite("imMatches.jpg", imMatches)

plt.figure(figsize=[15,10])
plt.imshow(imMatches[:,:,::-1])
plt.title("matched features form the two images")

points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt

# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
print("matrix \n{}".format(h))

# Use homography
im1Height, im1Width, channels = image1.shape
im2Height, im2Width, channels = image2.shape

im2Aligned = cv2.warpPerspective(image2, h, (im2Width + im1Width, im2Height))

plt.figure(figsize=[15,10])
plt.imshow(im2Aligned[:,:,::-1])
plt.title("The second image is aligned to the first one using homography and warping")

# Stitch Image 1 with aligned image 2
stitchedImage = np.copy(im2Aligned)
stitchedImage[0:im1Height,0:im1Width] = image1

plt.figure(figsize=[15,10])
plt.imshow(stitchedImage[:,:,::-1])
plt.title("**THE PANORAMIC IMAGE**")

