import os
from os.path import join as opj
import scipy.io as scio
import cv2
import numpy as np
import pickle
from scipy import spatial
import copy
import multiprocessing
from math import *
from sklearn import preprocessing
import tqdm

# Khoảng cách giữa các camera (6 cam)
# CAM_DIST = [[  0, 40, 55,100,120,145],
#             [ 40,  0, 15, 60, 80,105],
#             [ 55, 15,  0, 40, 65, 90],
#             [100, 60, 40,  0, 20, 45],
#             [120, 80, 65, 20,  0, 25],
#             [145,105, 90, 45, 25,  0]]

# 4 cam
CAM_DIST = [[  0, 40, 55,100],
            [ 40,  0, 15, 60],
            [ 55, 15,  0, 40],
            [100, 60, 40,  0]]

# SPEED_LIMIT = [[(0,0), (400,1300), (550,2000), (1000,2000), (1200, 2000), (1450, 2000)],
#                [(400,1300), (0,0), (100,900), (600,2000), (800,2000), (1050,2000)],
#                [(550,2000), (100,900), (0,0), (350,1050), (650,2000), (900, 2000)],
#                [(1000,2000), (600,2000), (350,1050), (0,0), (150,500), (450, 2000)],
#                [(1200, 2000), (800,2000), (650,2000), (150,500), (0,0), (250,900)],
#                [(1450, 2000), (1050,2000), (900, 2000), (450, 2000), (250,900), (0,0)]]  

# Speed_limit i- j: Khoảng frame tối thiểu - tối đa phương tiewnej đi từ cam i đén cam j
SPEED_LIMIT = [[(0,0), (400,1300), (550,2000), (1000,2000)],
               [(400,1300), (0,0), (100,900), (600,2000)],
               [(550,2000), (100,900), (0,0), (350,1050)],
               [(1000,2000), (600,2000), (350,1050), (0,0)]]  

rotate_270 = lambda i: [3,4,5,6,7,8,1,2,10][i % 10 - 1] # Xoay 270 độ 

# gán giá trị bằng 0 cho các tracklet thuộc cùng 1 cam (Không nhóm các tracklet trong cùng 1 cam với nhau)
def intracam_ignore(st_mask, cid_tids):
    count = len(cid_tids)
    for i in range(count):
        for j in range(count):
            if cid_tids[i][0] == cid_tids[j][0]:
                st_mask[i, j] = 0.
    return st_mask

# Lấy ra hướng 
def get_dire(zone_list,cid):
    zs,ze = zone_list[0],zone_list[-1]
    # Hướng zone_start - zone_end
    return (zs,ze)

# Lọc theo hướng đông - tây (do vị trí bố trí của camera , 2 hướng còn lại sẽ không xuất hiện ở camera cần theo dõi)
'''
Ví dụ:
cid_tids = (41,1)
zone_list = [2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 
            3, 3, 3, 3, 3, 3, 3, 3, 3]
frame_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
             26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 
             38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
             50, 51, 52, 53]
'''
def st_filter1(st_mask, cid_tids,cid_tid_dict):
    count = len(cid_tids)
    # west_in = {1,10}
    # west_out = {2,10}
    # east_in = {5,10}
    # east_out = {6,10}
    
    west = {3,0}
    east = {4,0}
    # lấy ra cặp i, j để tính toán giá trị phần tử trong ma trận
    for i in range(count):
        # Lấy ra tracklet có key là cid_tids[i]
        i_tracklet = cid_tid_dict[cid_tids[i]]
        # id cam
        i_cid = i_tracklet['cam']
        # starting_zone
        i_start = i_tracklet['zone_list'][0]
        # Ending_zone
        i_end = i_tracklet['zone_list'][-1]
        # Lấy ra các frame mà xe xuất hiện
        # Do vid có fps = 10 --> nhân với 10 để ra thời gian
        i_frame = [int(k * 10) for k in i_tracklet['io_time']]
        
        for j in range(count):
            j_tracklet = cid_tid_dict[cid_tids[j]]
            j_cid = j_tracklet['cam']
            j_start = j_tracklet['zone_list'][0]
            j_end = j_tracklet['zone_list'][-1]
            j_frame = [int(k * 10) for k in j_tracklet['io_time']]
            
            # Lấy ra giới hạn frame tối đa (có 4 cam: 41 - 44)
            frame_range = SPEED_LIMIT[i_cid-41][j_cid-41]
            
            # Khoảng thời gian cho phép di chuyển từ i - j
            forward_range = range(i_frame[1] + frame_range[0], i_frame[1] + frame_range[1])
            # Khoảng thời gian cho phép di chuyển từ j - i
            reverse_range = range(i_frame[0] - frame_range[1], i_frame[0] - frame_range[0])
            
            match = False
            # Đi từ hướng cam nhỏ sang lớn (đi hướng tây)
            if i_cid < j_cid:
                # i kết thúc ở tây và j bắt đầu ở đông và thời gian j xuất hiện trong khoảng foward_range
                if i_end in west and j_start in east and j_frame[0] in forward_range: # hướng tây
                    match = True
                # Ngược lại
                if i_start in west and j_end in east and j_frame[1] in reverse_range: # Lộn ngược về hướng đông
                    match = True
            # Đi từ cam có id lớn sang bé
            if i_cid > j_cid:
                # i kết thúc ở đông và j bắt đầu ở tây (W - E) và thời gian j xuất hiện trong khoảng reverse_range
                if i_start in east and j_end in west and j_frame[1] in reverse_range: # Lộn ngược về hướng tây
                    match = True
                # ngược lại
                if i_end in east and j_start in west and j_frame[0] in forward_range: # Hướng đông
                    match = True
        
            # Không khớp nhau thì gán giá trị tương đồng = 0 
            if not match:
                st_mask[i, j] = 0.0
                st_mask[j, i] = 0.0
    return st_mask

# # giảm thiểu không thời gian theo hướng (4 hướng)
# def st_filter(st_mask, cid_tids,cid_tid_dict):
#     count = len(cid_tids)
#     for i in range(count):
#         i_tracklet = cid_tid_dict[cid_tids[i]]
#         i_cid = i_tracklet['cam']
#         i_dire = get_dire(i_tracklet['zone_list'],i_cid)
#         i_iot = i_tracklet['io_time']
#         for j in range(count):
#             j_tracklet = cid_tid_dict[cid_tids[j]]
#             j_cid = j_tracklet['cam']
#             j_dire = get_dire(j_tracklet['zone_list'], j_cid)
#             j_iot = j_tracklet['io_time']

#             match_dire = True
#             cam_dist = CAM_DIST[i_cid-41][j_cid-41]
#             # THời gian trùng nhau
#             if i_iot[0]-cam_dist<j_iot[0] and j_iot[0]<i_iot[1]+cam_dist:
#                 match_dire = False
#             if i_iot[0]-cam_dist<j_iot[1] and j_iot[1]<i_iot[1]+cam_dist:
#                 match_dire = False

#             # Không phù hợp do i đi ra ngoài
#             if i_dire[1] in [1,2]: # i out
#                 if i_iot[0] < j_iot[1]+cam_dist:
#                     match_dire = False

#             if i_dire[1] in [1,2]:
#                 if i_dire[0] in [3] and i_cid>j_cid:
#                     match_dire = False
#                 if i_dire[0] in [4] and i_cid<j_cid:
#                     match_dire = False

#             if i_cid in [41] and i_dire[1] in [4]:
#                 if i_iot[0] < j_iot[1]+cam_dist:
#                     match_dire = False
#                 if i_iot[1] > 199:
#                     match_dire = False
#             if i_cid in [46] and i_dire[1] in [3]:
#                 if i_iot[0] < j_iot[1]+cam_dist:
#                     match_dire = False

       
#             if i_dire[0] in [1,2]:  # i in
#                 if i_iot[1] > j_iot[0]-cam_dist:
#                     match_dire = False

#             if i_dire[0] in [1,2]:
#                 if i_dire[1] in [3] and i_cid>j_cid:
#                     match_dire = False
#                 if i_dire[1] in [4] and i_cid<j_cid:
#                     match_dire = False

#             is_ignore = False
#             if ((i_dire[0] == i_dire[1] and i_dire[0] in [3, 4]) or (
#                     j_dire[0] == j_dire[1] and j_dire[0] in [3, 4])):
#                 is_ignore = True

#             if not is_ignore:
#                 # Xung đột hướng
#                 if (i_dire[0] in [3] and j_dire[0] in [4]) or (i_dire[1] in [3] and j_dire[1] in [4]):
#                     match_dire = False

#                 #Lọc trước khi chuyển cảnh
#                 if i_dire[1] in [3] and i_cid < j_cid:
#                     if i_iot[1]>j_iot[1]-cam_dist:
#                         match_dire = False
#                 if i_dire[1] in [4] and i_cid > j_cid:
#                     if i_iot[1]>j_iot[1]-cam_dist:
#                         match_dire = False

#                 if i_dire[0] in [3] and i_cid < j_cid:
#                     if i_iot[0]<j_iot[0]+cam_dist:
#                         match_dire = False
#                 if i_dire[0] in [4] and i_cid > j_cid:
#                     if i_iot[0]<j_iot[0]+cam_dist:
#                         match_dire = False
#                 ## ↑ 3-30

#                 ## 4-1
#                 if i_dire[0] in [3] and i_cid > j_cid:
#                     if i_iot[1]>j_iot[0]-cam_dist:
#                         match_dire = False
#                 if i_dire[0] in [4] and i_cid < j_cid:
#                     if i_iot[1]>j_iot[0]-cam_dist:
#                         match_dire = False
#                 ##

#                 # Lọc sau chuyển cảnh
#                 ## 4-7
#                 if i_dire[1] in [3] and i_cid > j_cid:
#                     if i_iot[0]<j_iot[1]+cam_dist:
#                         match_dire = False
#                 if i_dire[1] in [4] and i_cid < j_cid:
#                     if i_iot[0]<j_iot[1]+cam_dist:
#                         match_dire = False
#                 ##
#             else:
#                 if i_iot[1]>199:
#                     if i_dire[0] in [3] and i_cid < j_cid:
#                         if i_iot[0] < j_iot[0] + cam_dist:
#                             match_dire = False
#                     if i_dire[0] in [4] and i_cid > j_cid:
#                         if i_iot[0] < j_iot[0] + cam_dist:
#                             match_dire = False
#                     if i_dire[0] in [3] and i_cid > j_cid:
#                         match_dire = False
#                     if i_dire[0] in [4] and i_cid < j_cid:
#                         match_dire = False
#                 if i_iot[0]<1:
#                     if i_dire[1] in [3] and i_cid > j_cid:
#                             match_dire = False
#                     if i_dire[1] in [4] and i_cid < j_cid:
#                             match_dire = False

#             if not match_dire:
#                 st_mask[i, j] = 0.0
#                 st_mask[j, i] = 0.0
#     return st_mask

# phân cụm các cặp camera liên tiếp nhau theo vùng, đầu ra là cặp id camera thể hiện hướng đi 
def subcam_list(cid_tid_dict,cid_tids):
    # Quỹ đạo di chueyern cam i sang i + 1
    sub_3_4 = dict()
    # Quỹ đạo di chuyenr cam i sang i - 1
    sub_4_3 = dict()
    for cid_tid in cid_tids:
        # Lấy camid, tracklet id
        cid,tid = cid_tid
        # Lấy tất cả tracklet
        tracklet = cid_tid_dict[cid_tid]
        # starting zone, ending zone
        zs,ze = get_dire(tracklet['zone_list'], cid)
        if zs in [3] and cid not in [44]: # 4 to 3
            if not cid+1 in sub_4_3:
                sub_4_3[cid+1] = []
            sub_4_3[cid + 1].append(cid_tid)
        if ze in [4] and cid not in [41]: # 4 to 3
            if not cid in sub_4_3:
                sub_4_3[cid] = []
            sub_4_3[cid].append(cid_tid)
        if zs in [4] and cid not in [41]: # 3 to 4
            if not cid-1 in sub_3_4:
                sub_3_4[cid-1] = []
            sub_3_4[cid - 1].append(cid_tid)
        if ze in [3] and cid not in [44]: # 3 to 4
            if not cid in sub_3_4:
                sub_3_4[cid] = []
            sub_3_4[cid].append(cid_tid)

    sub_cid_tids = dict()
    for i in sub_3_4:
        sub_cid_tids[(i,i+1)]=sub_3_4[i]
    for i in sub_4_3:
        sub_cid_tids[(i,i-1)]=sub_4_3[i]
    return sub_cid_tids

def subcam_list2(cid_tid_dict,cid_tids):
    sub_dict = dict()
    for cid_tid in cid_tids:
        cid, tid = cid_tid
        if cid not in [41]:
            if not cid in sub_dict:
                sub_dict[cid] = []
            sub_dict[cid].append(cid_tid)
        if cid not in [46]:
            if not cid+1 in sub_dict:
                sub_dict[cid+1] = []
            sub_dict[cid+1].append(cid_tid)
    return sub_dict