import numpy as np
import pickle
import json
import os
from utils import set_dir

src_path = "../../datasets/algorithm_results/detect_merge/"

cam_names = ['c041','c042','c043','c044']
# cam_names = ['c041']

# tạo thư mục feature trong src, lưu kết quả vào file tên ảnh.json
def save_file(data,cam_name,image_name):
    save_path = os.path.join(src_path,cam_name)
    save_path = os.path.join(save_path,"feature")
    save_path = os.path.join(save_path,image_name+".json")
    with open(save_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder)
        
# chuyển dạng dự liệu thành json
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    # kiểm tra kiểu dữ liệu và chuyển đổi thành kiểu tiêu chuẩn của python
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Lưu dữ liệu từng camera theo từng frame
for cam_name in cam_names:
    # Tạo các thư mục con:
    cache_file = os.path.join(src_path,cam_name)
    # Lưu đặc trưng
    set_dir(os.path.join(cache_file,"feature"))
    # Kết quả detect
    set_dir(os.path.join(cache_file,"detect_result"))
    # Kết quả trung gian
    set_dir(os.path.join(cache_file,"mid_result"))
    # Kết quả quỹ đạo
    set_dir(os.path.join(cache_file,"traj_result"))
    # Kết quả theo dõi 1 track
    set_dir(os.path.join(cache_file,"track_result"))

    # Đọc dữ liệu từ tệp pickle chứa các thông tin của từng đối tượng
    cache_file = os.path.join(cache_file,cam_name + "_dets_feat.pkl")
    with open(cache_file, 'rb') as fid:
        data = pickle.load(fid)
        print(cache_file)
        # Lưu dữ liệu theo khung hình
        # print(data)
        frame_data = {}
        # Ảnh cuối cùng xử lý
        last_image_name = ""
        print(len(data))
        # Lấy từng dòng dữ liệu
        for index,key in enumerate(data):
            # print(key)
            # image name : tên của frame
            image_name = key.split("_")[0]
            # print(image_name)
            if index == 0:
                frame_data[key] = data[key]
                last_image_name = image_name
                continue
            if last_image_name != image_name:
                save_file(frame_data,cam_name,last_image_name)
                frame_data = {}
                last_image_name = image_name
            frame_data[key] = data[key]
            if index == len(data) -1:
                save_file(frame_data,cam_name,last_image_name)
    print('Done' + cam_name)

# print('Done pre-processing')

            
            