import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

file_path = 'E:\\AIC22-MCVT\\matching\\all_track.txt'
# # Test
# file_path = 'E:\\AIC22-MCVT\\datasets\\algorithm_results\\detect_merge\\c041\\res\\c041_mot.txt'
cam_path = defaultdict(list)

cam_path[41] = 'E:\\AIC22-MCVT\\datasets\\AIC22_Track1_MTMC_Tracking\\test\\S06\\c041\\vdo.avi'
cam_path[42] = 'E:\\AIC22-MCVT\\datasets\\AIC22_Track1_MTMC_Tracking\\test\\S06\\c042\\vdo.avi'
cam_path[43] = 'E:\\AIC22-MCVT\\datasets\\AIC22_Track1_MTMC_Tracking\\test\\S06\\c043\\vdo.avi'
cam_path[44] = 'E:\\AIC22-MCVT\\datasets\\AIC22_Track1_MTMC_Tracking\\test\\S06\\c044\\vdo.avi'

# print(cam_path)

f = open(file_path, 'r')
lines = f.readlines()

# Tách từng camera info
camInfo = defaultdict(list)
for i, line in enumerate(lines):
    camId, vehicleId, frameID, x, y, w, h, col1, col2 = parse_lines = line.strip().split()
    camId = int(camId)
    camInfo[camId].append([frameID, vehicleId, x, y, w, h])

all_cam_infor = []

# Tách từng thông tin từng frame với từng cam
for i in [41, 42, 43, 44]:
    camInfo[i] = sorted(camInfo[i], key = lambda x : int(x[0]))
    frameInfo = defaultdict(list)
    for line in camInfo[i]:
        frameID, vehicleId, x, y, w, h = line
        frameId = int(frameID)
        frameInfo[frameID].append([vehicleId, x, y ,w, h])
    all_cam_infor.append(frameInfo)
    # if i == 41:
    #     print('cam 41 frame 1')
    #     print(frameInfo['1'])

# print(len(all_cam_infor[0]))

def displayVid(vid_path, out_path, camName, numFrame):
    # Open vid
    cap = cv2.VideoCapture(vid_path)
    
    # Vid properties
    fourcc = cv2.VideoWriter_fourcc(*'AVI4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = 10  
    total_frame = numFrame
    
    # Create a VideoWriter object
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # # Test
    # f = open(det_path, 'r')
    # lines = f.readlines()
    
    frameCount = 0
    while(True):
        ret, frame = cap.read()
        frameCount = frameCount + 1
        if not(ret):
            break
        
        # for line in lines:
        #     # print(line)
        #     frameid, vehicleid, x, y, w, h, conf, col1, col2, col3 = line.split(',')
        #     frameid = int(frameid)
        #     x = int(float(x))
        #     y = int(float(y))
        #     w = int(float(w))
        #     h = int(float(h))
        #     if frameid != frameCount - 1: continue
        #     # Draw bounding box
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        #     # Put the vehicle ID near the bounding box
        #     cv2.putText(frame, f'ID: {vehicleid}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
        #                 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        for obj in all_cam_infor[camName - 41][str(frameCount)]:
            # print('????')
            idVehicle, x, y, w, h = obj
            # print(x,y,w, h)
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put the vehicle ID near the bounding box
            cv2.putText(frame, f'ID: {idVehicle}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 0, 0), 2, cv2.LINE_AA)
        #     # print(obj)
        
        cv2.imshow(str(camName), frame)
        out.write(frame)
        
        time.sleep(1/fps)
        
        if frameCount == total_frame:
            break
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Test
    # in_path = Path(cam_path[41])
    # out_path = Path('E:\\AIC22-MCVT\\genvid\\test.mp4')
    # det_path = Path('E:\\AIC22-MCVT\\datasets\\algorithm_results\\detect_merge\\c041\\res\\c041_mot.txt')
    # displayVid(in_path, out_path, 999, 400, det_path)
    for camid in [41, 42, 43, 44]:
        out_path = Path('E:\\AIC22-MCVT\\genvid\\' + 'cam'+str(camid) + '.mp4')
        in_path = Path(cam_path[camid])
        displayVid(in_path, out_path, camid, 400)
