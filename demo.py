import argparse
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2
from PIL import Image, ImageDraw
from models.pfld_vovnet import vovnet_pfld
from models.pfld import PFLDInference
from mtcnn.detector import detect_faces, show_bboxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['plfd_backbone'])
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    im = Image.open(args.image_path)
    img = np.array(im)
    height, width = img.shape[:2]
    draw = ImageDraw.Draw(im)
    bounding_boxes, landmarks = detect_faces(img)
    print(bounding_boxes)
    for box in bounding_boxes:
        score = box[4]
        x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        
        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0).to(device)
        landmarks = pfld_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]
        print(pre_landmark)
        for (x, y) in pre_landmark.astype(np.int32):
#            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))
            draw.ellipse((x1+x-1,y1+y-1,x1+x+1,y1+y+1), fill=(0,0,255))

    im.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./checkpoint/pfld_weight.pth.tar",
        type=str)
    parser.add_argument('--image_path', default="demo_img.jpg", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
