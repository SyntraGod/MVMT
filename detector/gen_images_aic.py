import os
import sys
sys.path.append('../')
from config import cfg
import cv2
from tqdm import tqdm

def preprocess(src_root, dst_root):
    # print(src_root)
    
    # Kiêm tra src_root có tồn tại không
    if not os.path.isdir(src_root):
        print("[Err]: invalid source root")
        return
    
    # Nếu dst_root chưa tồn tại thì tạo thư mục
    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    sec_dir_list = ['test']
    dst_dir_list = [ dst_root + '/images/' + i for i in sec_dir_list]
    
    # sec_dir_list = 'test'
    # dst_dir_list = '/AIC21-MTMC/datasets/algorithm_results/detection//images/test'
    
    for i in dst_dir_list:
        if not os.path.isdir(i):
            os.makedirs(i)
            
    for i,x in enumerate(sec_dir_list):
        x_path = src_root + '/' + x
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                if y.startswith('S'):
                    y_path = os.path.join(x_path,y)
                    # y_path = '/AIC21-MTMC/datasets/AIC22_Track1_MTMC_Tracking//test/S06'
                    
                    # So cam
                    cur_cam = 0
                    # Chi lay 4 cam
                    max_cam = 4  
                    
                    # Lay tat ca camera trong thu muc (6 cam) 
                    for z in os.listdir(y_path):
                        if cur_cam == max_cam: break
                        z_path = os.path.join(y_path,z)
                        print(z_path)
                        if z.startswith('c'):
                            cur_cam = cur_cam + 1
                            video_path = os.path.join(z_path,'vdo.avi')
                            roi_path = os.path.join(z_path, 'roi.jpg')
                            ignor_region = cv2.imread(roi_path)

                            dst_img1_dir = os.path.join(dst_dir_list[i],y,z,'img1')
                            if not os.path.isdir(dst_img1_dir):
                                os.makedirs(dst_img1_dir)

                            video = cv2.VideoCapture(video_path)
                            # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                            # Chi lay 300 frame dau cua video
                            frame_count = 400
                            frame_current = 0
                            while frame_current<frame_count-1:
                                frame_current = int(video.get(cv2.CAP_PROP_POS_FRAMES))
                                _, frame = video.read()
                                dst_f =  'img{:06d}.jpg'.format(frame_current)
                                dst_f_path = os.path.join(dst_img1_dir , dst_f)
                                if not os.path.isfile(dst_f_path):
                                    frame = draw_ignore_regions(frame, ignor_region)
                                    cv2.imwrite(dst_f_path, frame)
                                    print('{}:{} generated to {}'.format(z,dst_f, dst_img1_dir))
                                else:
                                    print('{}:{} already exists.'.format(z,dst_f))

# Loai bo cac vung khong can thiet
def draw_ignore_regions(img, region):
    if img is None:
        print('[Err]: Input image is none!')
        return -1
    img = img*(region>0)

    return img
if __name__ == '__main__':
    path = cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    save_dir = cfg.DET_SOURCE_DIR.split('images')[0]
    preprocess(src_root=f'{cfg.CHALLENGE_DATA_DIR}',
               dst_root=f'{save_dir}')
    print('Done')
