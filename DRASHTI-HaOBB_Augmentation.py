import cv2
import numpy as np
import random
import os
import argparse

degrees= 180
translate= 0.1
scale= 0.5
hsv_h= 0.0188
hsv_s= 0.704
hsv_v= 0.36

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed
        
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class_dict = {
    "Auto3WCargo": 0,
    "AutoRicksaw": 1,
    "Bus": 2,
    "Container": 3,
    "Mixer": 4,
    "MotorCycle": 5,
    "PickUp": 6,
    "SUV": 7,
    "Sedan": 8,
    "Tanker": 9,
    "Tipper": 10,
    "Trailer": 11,
    "Truck": 12,
    "Van": 13
}
colors = Colors()  # create instance for 'from utils.plots import colors'

#label format: (number of gts,[cls, x1,y1,x2,y2,x3,y3,x4,y4, θ])
def read_label(lb_file):
    if os.path.isfile(lb_file):  
        with open(lb_file) as f:
            labels = []
            for line in f.read().strip().splitlines():
                if not line.strip():
                    continue  # skip empty lines
                
                if line.startswith("flightHeight"):
                    continue  # skip metadata
                
                parts = line.split()
                
                if len(parts) < 11:
                    continue  # skip invalid lines safely
                
                labels.append(parts)           
        l_ = []
        for label in labels:
            if label[-1] == "2": # diffcult
                continue
            
            cls_id = class_dict[label[8]]
            coords = np.array(label[:8], dtype=np.float32)
            
            l_.append(np.concatenate((cls_id, coords, float(label[10])), axis=None))
        l = np.array(l_, dtype=np.float32)
        return l
    else:
        return None

def apply_aug(img, labels, flip_ud=False, flip_lr=False, hsv=False, affine=False):
    nl=len(labels)
    img_h, img_w = img.shape[0], img.shape[1]

    #color augmentation
    if nl and hsv:
        augment_hsv(img, hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)
        
    #vertical flip: up-down 
    if nl and flip_ud:
        img = np.flipud(img)
        labels[:, 2:-1:2] = img_h - labels[:, 2:-1:2] - 1
        labels[:, -1] = (-labels[:, -1]) % 360

    #horizontal flip: left-right 
    if nl and flip_lr:
        img = np.fliplr(img)
        labels[:, 1:-1:2] = img_w - labels[:, 1:-1:2] - 1
        labels[:, -1] = (180 - labels[:, -1]) % 360

    
    #afine transformation
    if nl and affine:
        angle = random.uniform(-degrees, degrees)  # CCW positive
        scale_factor = random.uniform(1 - scale, 1 + scale)

        tx = random.uniform(-translate, translate) * img_w
        ty = random.uniform(-translate, translate) * img_h

        # -----------------------
        # 2. Build affine matrix
        # -----------------------
        center = (img_w / 2, img_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale_factor)

        # Add translation
        M[:, 2] += (tx, ty)

        # -----------------------
        # 3. Transform image
        # -----------------------
        img = cv2.warpAffine(
            img,
            M,
            dsize=(img_w, img_h),
            flags=cv2.INTER_LINEAR,
            borderValue=(114, 114, 114),
            )
        # reshape polygon points
        polys = labels[:, 1:9].reshape(-1, 4, 2)

        # add ones for affine
        ones = np.ones((polys.shape[0], 4, 1))
        polys_homo = np.concatenate([polys, ones], axis=2)  # Nx4x3

        # apply affine transform
        polys_transformed = np.matmul(polys_homo, M.T)

        labels[:, 1:9] = polys_transformed.reshape(-1, 8)

        # -----------------------
        # 5. Update heading angle
        # -----------------------
        # since image rotated by `angle` CCW
        # and your theta is CW positive
        labels[:, -1] = (labels[:, -1] - angle) % 360
        
    img = np.ascontiguousarray(img)
    return img, labels

   
def draw_lbl(img, labels, out_file):
    for label in labels:
        #print(label)
        #label format: (number of gts,[cls, x1,y1,x2,y2,x3,y3,x4,y4, θ])        
        color = colors(label[0])
        poly=label[1:9]
        pts = np.array(poly, dtype=np.float32).reshape(4, 2)
        # OpenCV min area rect
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        if w < h:
            w, h = h, w
       
        theta_rad = np.deg2rad(label[9])
        dx = (w / 2) * np.cos(theta_rad)
        dy = (w / 2) * np.sin(theta_rad)
        x2 = int(cx + dx)
        y2 = int(cy + dy)
        cx = int(cx)
        cy = int(cy)
        polygon_list = np.array([(poly[0], poly[1]), (poly[2], poly[3]),(poly[4], poly[5]), (poly[6], poly[7])], np.int32)
        cv2.drawContours(img, contours=[polygon_list], contourIdx=-1, color=color, thickness=2)
        cv2.arrowedLine(img, pt1=(cx,cy), pt2=(x2,y2), color=color, thickness=2, tipLength=0.25)
    cv2.imwrite(out_file,img)
    return



def parse_opt():
    parser = argparse.ArgumentParser()
    #take image and lable as inputs
    parser.add_argument('--img', type=str, required=True, help='path to input image')
    parser.add_argument('--lbl', type=str, required=True, help='path to respective label text file in DRASHTI-HaOBB format')
    
    # Augmentation switches (all default False)
    parser.add_argument('--flipud', action='store_true', help='enable vertical flip augmentation')
    parser.add_argument('--fliplr', action='store_true', help='enable horizontal flip augmentation')
    parser.add_argument('--hsv', action='store_true', help='enable HSV augmentation')
    parser.add_argument('--affine', action='store_true', help='enable affine augmentation')

    return parser.parse_args()



def main():
    opt = parse_opt()
    img_file=opt.img
    lb_file=opt.lbl
    labels=read_label(lb_file)   
    if labels.shape[0]:
        img=cv2.imread(img_file)
        img,labels=apply_aug(img, labels, flip_ud= opt.flipud, flip_lr= opt.fliplr, hsv=opt.hsv, affine= opt.affine)
        out_file="augmented_image.jpg"
        draw_lbl(img, labels, out_file)
    
if __name__ == "__main__":
    main()
