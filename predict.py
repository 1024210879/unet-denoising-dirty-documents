import cv2
import numpy as np
import torch
import os
from unet import UNet

weight = './weight/weight.pth'

# load net
print('load net')
net = UNet(1, 2)
if os.path.exists(weight):
    checkpoint = torch.load(weight)
    net.load_state_dict(checkpoint['net'])
else:
    exit(0)

# load img
print('load img')
dir = './data/test/'
filenames = [os.path.join(dir, filename)
             for filename in os.listdir(dir)
             if filename.endswith('.jpg') or filename.endswith('.png')]
totalN = len(filenames)

for index, filename in enumerate(filenames):
    img = cv2.imread(filename, 0)
    if img is None:
        print('img is None')
        continue
    h, w = img.shape[0], img.shape[1]
    while h % 4 != 0:
        h += 1
    while w % 4 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    # img = cv2.blur(img, (3, 3))

    input = torch.from_numpy(img[np.newaxis][np.newaxis]).float() / 255
    output = net(input)[0, 0].detach().numpy()
    res = np.concatenate((img/255, output), axis=1)

    winname = filename + ' %d | %d ' % (index + 1, totalN)
    cv2.imshow(winname, res)
    cv2.waitKey()
    cv2.destroyWindow(winname)
