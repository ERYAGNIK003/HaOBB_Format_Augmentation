import xml.etree.ElementTree as ET
import math
import numpy as np
import cv2
import os
import shutil

#Update images and XML path
img_path="imgs"
xml_path="annotations.xml"
##################


def get_rotated_box_points(xtl, ytl, xbr, ybr, angle_deg):
    isTouching=False
    # center, width, height
    cx = (xtl + xbr) / 2
    cy = (ytl + ybr) / 2
    w = xbr - xtl
    h = ybr - ytl

    angle = math.radians(angle_deg)

    # rectangle corners (before rotation)
    dx = w / 2
    dy = h / 2
    corners = [
        (-dx, -dy),
        ( dx, -dy),
        ( dx,  dy),
        (-dx,  dy)
    ]

    # rotate around center
    points = []
    for x, y in corners:
        xr = x * math.cos(angle) - y * math.sin(angle) + cx
        yr = x * math.sin(angle) + y * math.cos(angle) + cy
        if (xr>=3835 or xr<=5) or (yr>=2155 or yr<=5):
            isTouching=True
        points.append((xr, yr))

    return cx,cy,points,isTouching  # 4 points

def get_frame_to_image_map(image_dir):
    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg'))
    ])
    print("Total images are ",len(images))
    return images  # index = frame number


def rotateCart(origin, point, angle):
    ox, oy = origin
    px, py = point
    oy=-oy
    py=-py

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    qy=-qy
    return int(qx), int(qy)

def read_cvat_video_tracks(xml_file,image_dir,framewiseData,vidname):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    tracks_data = []
    images=get_frame_to_image_map(image_dir)
    
    for track in root.findall("track"):
        track_id = int(track.get("id"))
        label = track.get("label")

        track_info = {
            "track_id": track_id,
            "label": label,
            "frames": []
        }

        for box in track.findall("box"):
            outside = int(box.get("outside"))
            if outside == 1:
                continue
            frame = int(box.get("frame"))
            rotation_str=box.get("rotation")
            angle = float(rotation_str) if rotation_str is not None else 0.0
            
            #angle = float(box.get("rotation"))

            #print(f"track {track_id} frame {frame} angle {angle}")
            
            image_name = images[frame]
            image_id = os.path.splitext(image_name)[0]
            
            xtl = float(box.get("xtl"))    
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            cx,cy,points,isTouching = get_rotated_box_points(
                xtl, ytl, xbr, ybr, angle
            )
            pts = np.array(points, dtype=float)
            xmin = pts[:, 0].min()
            ymin = pts[:, 1].min()
            xmax = pts[:, 0].max()
            ymax = pts[:, 1].max()
            xtl1 = max(0, min(1920, xmin))
            xbr1 = max(0, min(1920, xmax))
            ytl1 = max(0, min(1080, ymin))
            ybr1 = max(0, min(1080, ymax))
            crop_reg=[int(xtl1),int(ytl1),int(xbr1),int(ybr1)]
            #cen=[int((xtl1+xbr1)/2),int((ytl1+ybr1)/2)]
            cx,cy=int((xtl1+xbr1)/2),int((ytl1+ybr1)/2)
            w2=(xbr1-xtl1)/2.0
            cen=[cx,cy]
            track_info["frames"].append({
                "frame": frame,
                "imagename":image_id,
                "angle": angle,
                "cen":cen,
                "isTouching":isTouching,
                "points": points  # 4 (x,y)
                
            })

            up_angCW=0
            up_id=track_id+1
            
                
            hood_pt=[cx+w2,cy]
            hood_arrow=[[cx,cy],hood_pt]
            hood_arrow_rot = [rotateCart((cx,cy), pt, math.radians(-angle)) for pt in hood_arrow]
        
            if image_id in framewiseData:
                framewiseData[image_id].append([up_id,label,points,angle,cen,hood_arrow_rot,crop_reg,isTouching])
            else:
                framewiseData[image_id]=[[up_id,label,points,angle,cen,hood_arrow_rot,crop_reg,isTouching]]
        tracks_data.append(track_info)

    return tracks_data




vids=[img_path]

outDir=os.path.join(".","output")
os.makedirs(outDir, exist_ok=True)
if os.path.exists(outDir) and os.path.isdir(outDir):
    shutil.rmtree(outDir)
    print("Removed all previous data")
os.makedirs(outDir, exist_ok=True)
clsCNTs={}

for vidname in vids:
    print("Working on : ",vidname)
    vidDir=outDir#os.path.join(outDir,vidname)
    os.makedirs(vidDir, exist_ok=True)
    
    
    imgPath=os.path.join(vidname)

    framewiseData={}
    tracks = read_cvat_video_tracks(xml_path,imgPath,framewiseData,vidname)

    annDir=os.path.join(vidDir,"annotated")
    dotaDir=os.path.join(vidDir,"drashti_lbls")
    os.makedirs(annDir, exist_ok=True)
    os.makedirs(dotaDir, exist_ok=True)
    
    for frm,annotations in framewiseData.items():
        #print("Working on ",frm,len(annotations))
        img=cv2.imread(os.path.join(imgPath,frm+".jpg"))
        dota_lines=[f"flightHeight:{0}\n"]
        for ann in annotations:
            track_id,cls,points,angCW,cen,hood,crop_reg,isTouching=ann[0],ann[1],ann[2],ann[3],ann[4],ann[5],ann[6],ann[7]
            
            if cls in clsCNTs:
                clsCNTs[cls]+=1
            else:
                clsCNTs[cls]=1
            ####
            
            pts = np.array(points, dtype=np.int32)
            # ---- write DOTA line ----
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            x3, y3 = pts[2]
            x4, y4 = pts[3]
            difficulty = 0  # you can change later
            if isTouching:
                difficulty=1
                
            dota_lines.append(f"{x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {cls} {difficulty} {angCW}\n")
            cv2.arrowedLine(img, hood[0], hood[1], (0, 0, 255), 2, tipLength=0.25)
            if isTouching:
                cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            else:
                cv2.polylines(img, [pts], isClosed=True, color=(0,0, 0), thickness=3)
            
        cv2.imwrite(os.path.join(annDir,frm+".jpg"),img)
        txt_path = os.path.join(dotaDir, f"{int(frm)}.txt")
        with open(txt_path, "w") as f:
            for line in dota_lines:
                f.write(line)
                
print("Classwise counts : ",clsCNTs)
    
