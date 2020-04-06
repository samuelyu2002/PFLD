from PIL import Image
import cv2
import numpy as np
import random
from torchvision import transforms
def channel_shuffle(img):
    if(img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img


def random_noise(img, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H,W)) * 255

        img = img + noise[:,:,np.newaxis]*np.array([1,1,1])
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    return img

def random_brightness(img, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * image
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_contrast(img, contrast=0.3):
    coef = np.array([[[0.114, 0.587,  0.299]]])   # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha*img  + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_saturation(img, saturation=0.5):
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray  = img * coef
    gray  = np.sum(gray,axis=2, keepdims=True)
    img = alpha*img  + (1.0 - alpha)*gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def random_hue(image, hue=-0.1):
#    h = int(np.random.uniform(-hue, hue)*180)
    h = hue*180
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

#img = cv2.imread('/home/mv01/PFLD/data/300W_newer/train_data/imgs/3_image_0833_11.png')
img = Image.open('/home/mv01/PFLD/data/300W_newer/train_data/imgs/3_image_0833_11.png')
#img = np.asarray(img)
img.show()
transform= transforms.Compose([transforms.ColorJitter(0.2,0.2,0.2,0.05)]) #0.2,0.2,0.2
img = transform(img)
img.show()
#Image.fromarray(img[...,(2,1,0)]).show()
#img = random_noise(img)
#Image.fromarray(img[...,(2,1,0)]).show()
