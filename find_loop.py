import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='../rgbd_dataset/RGBD/')
parser.add_argument('--dataset_name', type=str, default='Nami7')
parser.add_argument('--start', type=int, default=12)
parser.add_argument('--threshold', type=int, default=500)
args = parser.parse_args()

dataset_dir = args.dataset_dir + '{}/'.format(args.dataset_name)
rgb_dir = dataset_dir + 'rgb/'

# get dataset
rgb_files = glob.glob(rgb_dir + '*.png')
numbers = [int(r.split('/')[-1].split('.')[0]) for r in rgb_files]
sorted_index = np.argsort(numbers)
rgb_files = [rgb_files[idx] for idx in sorted_index]
numbers = [numbers[idx] for idx in sorted_index]

start = args.start

rgb_files = rgb_files[start:]
numbers = numbers[start:]

start_num = numbers[0]

target_img = cv2.imread(rgb_files[0])

# set detector
detector = cv2.ORB_create(2000)  # nfeatures = 2000

# matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

target_kp, target_des = detector.detectAndCompute(target_img, None)

threshold = args.threshold

if len(target_kp) < threshold:
    print('Please set another proper value!!')
    exit()

target_pts = np.array([c.pt for c in target_kp[:threshold]])

# create target image with feature points
for pt in target_pts:
    x, y = pt
    x, y = int(x), int(y)
    target_img = cv2.circle(target_img,(x, y), 2, (0,0,255), -1)


# required variables
nearest_idx = start
min_loss = np.inf

# skip nearest frame
skip = 10

for num, rgb_path in zip(numbers[skip:], rgb_files[skip:]):
    cur_img = cv2.imread(rgb_path)
    cur_kp, cur_des = detector.detectAndCompute(cur_img, None)
    # print('kps num:', len(cur_kp))
    
    if len(cur_kp) < threshold:
        continue
    
    cur_pts = np.array([c.pt for c in cur_kp[:threshold]])
    
    cur_loss = np.sum(np.abs(cur_pts - target_pts)) / threshold
    
    print('current loss:', cur_loss)
    
    
    for pt in cur_pts:
        x, y = pt
        x, y = int(x), int(y)
        cur_img = cv2.circle(cur_img,(x, y), 2, (0,0,255), -1)
    
    """
    fig, ax = plt.subplots(1, 2)    
    ax[1].imshow(cur_img)
    ax[0].imshow(target_img)
    ax[1].set_title('current')
    ax[0].set_title('target')
    plt.show()
    """
    
    
    if cur_loss < min_loss:
        min_loss = cur_loss
        nearest_idx = num
        ans_img = cur_img

print('start:', start_num)
print('ans:', nearest_idx)

# describe result
fig, ax = plt.subplots(1, 2)    
ax[1].imshow(cur_img)
ax[0].imshow(target_img)
ax[1].set_title('current')
ax[0].set_title('target')
plt.show()  
        
    



