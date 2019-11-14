import cv2
import numpy as np
import random


class ToGray(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(label.shape) == 3:
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        return image, label


class RondomFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        degree = random.random()
        if degree <= 0.33:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        elif degree <= 0.66:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


class RandomRotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        angle = random.random()*self.angle
        angle = angle if random.random() < 0.5 else -angle
        h, w = image.shape[0], image.shape[1]
        scale = random.random()*0.4 + 0.9
        matRotate = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
        image = cv2.warpAffine(image, matRotate, (w, h))
        label = cv2.warpAffine(label, matRotate, (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return image, label


class RandomCrop(object):
    def __init__(self, crop_h, crop_w):
        self.crop_h, self.crop_w = crop_h, crop_w

    def __call__(self, image, label):
        h, w = image.shape[0], image.shape[1]
        crop_x = int(random.random() * (w - self.crop_w))
        crop_y = int(random.random() * (h - self.crop_h))
        image = image[crop_y: crop_y + self.crop_h, crop_x: crop_x + self.crop_w]
        label = label[crop_y: crop_y + self.crop_h, crop_x: crop_x + self.crop_w]
        return image, label


class EqualizeHist(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, label):
        if random.random() < self.degree:
            image = cv2.equalizeHist(image)
        return image, label


class Blur(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, label):
        if random.random() < self.degree:
            image = cv2.blur(image, (3, 3))
        return image, label


class Log(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, image, label):
        if random.random() < self.degree:
            image = np.log(1 + image.astype(np.float32)/255) * 255
        return image.astype(np.uint8), label


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, label):
        image = image / 255
        label = (255 - label) // 50
        label[label > 1] = 1
        return image.astype(np.float32), label.astype(np.int64)


if __name__ == '__main__':
    transforms = [
        ToGray(),
        RondomFlip(),
        RandomRotate(15),
        RandomCrop(128, 128),
        Log(0.5),
        EqualizeHist(0.5),
        Blur(0.5),
        ToTensor()
    ]
    trans_len = len(transforms)
    image = cv2.imread('./data/train/2.png')
    label = cv2.imread('./data/train_cleaned/2.png')
    for i in range(1):
        img1, img2 = image, label
        for index, transform in enumerate(transforms):
            img1, img2 = transform(img1, img2)
            img = np.concatenate((img1, img2), axis=1)
            cv2.imshow("%d | %d" % (index + 1, trans_len), img)
            cv2.waitKey()
            cv2.destroyAllWindows()
