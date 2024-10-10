from utils.filter import *
from utils.visual_rr import visual_rerank
from sklearn.cluster import AgglomerativeClustering
import sys, copy
sys.path.append('../')
from config import cfg

# Ma trận tương đồng
def get_sim_matrix(_cfg,cid_tid_dict,cid_tids, sub_c_to_c):
    count = len(cid_tids)
    print('count: ', count)

    # đặc trưng mean_feat của các đối tượng cid_tid_dict
    q_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    g_arr = np.array([cid_tid_dict[cid_tids[i]]['mean_feat'] for i in range(count)])
    q_arr = normalize(q_arr, axis=1)
    g_arr = normalize(g_arr, axis=1)
    # sim_matrix = np.matmul(q_arr, g_arr.T)
    
    # print("P,Q ARRAY")
    # print(q_arr)
    # print(g_arr)

    # st mask: mặt nạ không thời gian, giá trị 0 - 1: không có khả năng khớp / có khả năng khớp
    st_mask = np.ones((count, count), dtype=np.float32)
    
    # Gán độ tương đồng của các tracklet cùng camera = 0
    st_mask = intracam_ignore(st_mask, cid_tids)
    
    # giảm thiểu độ tương đồng trên tiêu chí không-thời gian
    st_mask = st_filter1(st_mask, cid_tids, cid_tid_dict)
    # print('st mask')
    # print(st_mask)

    # TÍnh toán tương đồng dựa trên cosine
    # visual rerank
    visual_sim_matrix = visual_rerank(q_arr, g_arr, cid_tids, _cfg)
    # print('visual_sim_matrix')
    # print(visual_sim_matrix)
    visual_sim_matrix = visual_sim_matrix.astype('float32')    

    # merge result
    # DOdoongf bộ với mặt nạ không thời gian
    np.set_printoptions(precision=3)
    sim_matrix = visual_sim_matrix * st_mask

    # sim_matrix[sim_matrix < 0] = 0
    # Loại bỏ tương đồng của đối tượng với bản thân nó
    np.fill_diagonal(sim_matrix, 0)
    return sim_matrix

# Chuẩn hóa
def normalize(nparray, axis=0):
    nparray = preprocessing.normalize(nparray, norm='l2', axis=axis)
    return nparray

# Nhóm các phần tử cùng label lại với nhau
def get_match(cluster_labels):
    '''
    cluster_labels = [36 31 35 34 33 32 15 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16  7 14
                     13 12 11 10  9  8  3  6  5  4  1  2  0]
    '''
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        # Nếu nhãn l đã trong cụm, thêm chỉ số i  của phần tử vào cluster_dict[l]
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            # Nếu chưa tạo 1 danh sách mới có 1 nhãn là i
            cluster_dict[l] = [i]
    # Tạo danh sách chứa tất cả cacs cụm
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    # print(cluster)
    return cluster

# ánh xạ nhãn phân cụm với danh sách tracklet ID
def get_cid_tid(cluster_labels,cid_tids):
    cluster = list()
    for labels in cluster_labels:
        cid_tid_list = list()
        for label in labels:
            cid_tid_list.append(cid_tids[label])
        cluster.append(cid_tid_list)
    return cluster

def combin_cluster(sub_labels,cid_tids):
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_cluster2(sub_labels,cid_tids):
    sub_labels_copy = copy.deepcopy(sub_labels)
    cluster = list()
    for sub_c_to_c in sub_labels:
        if len(cluster)<1:
            cluster = sub_labels[sub_c_to_c]
            continue

        for c_ts in sub_labels[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break
            if not is_add:
                cluster.append(c_ts)
    # print(cluster)

    for sub_c_to_c in sub_labels_copy:
        for c_ts in sub_labels_copy[sub_c_to_c]:
            is_add = False
            for i_c, c_set in enumerate(cluster):
                if len(set(c_ts)-set(c_set)) !=0 and len(set(c_ts) & set(c_set))>0:
                    new_list = list(set(c_ts) | set(c_set))
                    cluster[i_c] = new_list
                    is_add = True
                    break

    labels = list()
    num_tr = 0
    for c_ts in cluster:
        label_list = list()
        for c_t in c_ts:
            label_list.append(cid_tids.index(c_t))
            num_tr+=1
        label_list.sort()
        labels.append(label_list)
    print("new tricklets:{}".format(num_tr))
    return labels,cluster

def combin_feature(cid_tid_dict,sub_cluster):
    for sub_ct in sub_cluster:
        if len(sub_ct)<2: continue
        mean_feat = np.array([cid_tid_dict[i]['mean_feat'] for i in sub_ct])
        for i in sub_ct:
            cid_tid_dict[i]['mean_feat'] = mean_feat.mean(axis=0)
    return cid_tid_dict

# Phân cụm các labels
'''
_cfg: chứa trong file config/defaults
cid_tid_dict: chứa quỹ đạo của các phương tiện trên từng cam. Key = (camid, trackletid)
cid_tids: chứa các tuple (cid, tid) đã được sắp xếp theo từ điển
score_thr = 0.5
'''
def get_labels(_cfg, cid_tid_dict, cid_tids, score_thr):
    # 1st cluster: phân cụm theo vùng 2 camera liên tiếp nhau
    sub_cid_tids = subcam_list(cid_tid_dict,cid_tids)
    
    # print("list cụm theo vùng")
    # print(sub_cid_tids)
    '''
    {(41, 42): [(41, 1), (41, 3), (41, 5), (41, 9), (41, 11), (41, 13), (41, 17), (41, 20), 
                (41, 22), (41, 26), (41, 39), (41, 42), (41, 44), (42, 2), (42, 3), (42, 4), 
                (42, 5), (42, 7), (42, 8), (42, 9), (42, 13), (42, 14), (42, 16), (42, 19), 
                (42, 21), (42, 22), (42, 23), (42, 24), (42, 25), (42, 26), (42, 27), (42, 28), 
                (42, 29), (42, 31), (42, 32), (42, 36), (42, 39)],
    (42, 43):  [(42, 5), (42, 7), (42, 8), (42, 13), (42, 14), (42, 19), (42, 22), (42, 23), 
                (42, 25), (42, 26), (42, 29), (42, 31), (42, 32), (42, 35), (42, 36), (42, 39), 
                (43, 4), (43, 5), (43, 6), (43, 8), (43, 19)], 
    (43, 44):  [(43, 7), (43, 9), (44, 1), (44, 2), (44, 3), (44, 4), (44, 7), (44, 10), (44, 13), 
                (44, 14), (44, 21), (44, 23), (44, 29), (44, 30), (44, 74)], 
    (42, 41):  [(41, 2), (41, 27), (41, 30), (41, 48), (42, 2), (42, 15)], 
    (43, 42):  [(42, 6), (42, 15), (43, 4), (43, 5), (43, 6), (43, 8), (43, 21)], 
    (44, 43):  [(44, 1), (44, 2), (44, 3), (44, 4), (44, 8), (44, 11), (44, 13), (44, 18), (44, 20), 
                (44, 24), (44, 31), (44, 56), (44, 61), (44, 63), (44, 71), (44, 76), (44, 81)]}
    '''
    
    sub_labels = dict()
    # Ngưỡng tương đồng
    dis_thrs = [0.7,0.5,0.5,0.5,0.5,
                0.7,0.5,0.5,0.5,0.5]
    # print(len(sub_cid_tids))
    
    # sub_c_to_c : các cặp camera liền kề
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        # TÍnh ma trận tương đồng giữa các track của 2 camera kế tiếp
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict,sub_cid_tids[sub_c_to_c], sub_c_to_c)
        
        # print(sim_matrix)
        # Phân cụm phân cấp, dựa trên ngưỡng khoảng cách distance_threshold
        # Sử dụng 1 - sim_matrix vì sim_matrix là ma trận tương đồng, tương đồng càng cao -> khoảng cách càng thâops
        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        # print("CLuster labels")
        # print(cluster_labels)
        '''
        cluster_labels = [36 31 35 34 33 32 15 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16  7 14
                        13 12 11 10  9  8  3  6  5  4  1  2  0]
        '''
        labels = get_match(cluster_labels)
        # print('Labels')
        # print(labels)
        '''
        labels = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], 
                 [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], 
                 [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], 
                 [34], [35], [36]]
        '''
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        # print(cluster_cid_tids)
        sub_labels[sub_c_to_c] = cluster_cid_tids
       
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster2(sub_labels,cid_tids)

    u_turn = []
    for cluster in sub_cluster:
        temp_set = set()
        for tl in cluster:
            if tl[0] not in temp_set:
                temp_set.add(tl[0])
            else:
                turn_cam = tl[0]
                u_turn.append([i for i in cluster if i[0] == turn_cam])

    # 2nd cluster
    cid_tid_dict_new = combin_feature(cid_tid_dict, sub_cluster)
    sub_cid_tids = subcam_list2(cid_tid_dict_new,cid_tids)
    sub_labels = dict()
    dis_thrs = [0.11,0.1,0.12,0.1,0.27]
    for i,sub_c_to_c in enumerate(sub_cid_tids):
        sim_matrix = get_sim_matrix(_cfg,cid_tid_dict_new,sub_cid_tids[sub_c_to_c], sub_c_to_c)

        tmp_list = list()
        a = (sim_matrix < 0.3) & (sim_matrix > 0.1)
        index = np.where(a)
        for k in range(len(index[0])):
            tmp_list.append(set([sub_cid_tids[sub_c_to_c][index[0][k]],sub_cid_tids[sub_c_to_c][index[1][k]]]))

        for tls in u_turn:
            try:
                tl1, tl2 = tls[0], tls[1]
                idx1 = sub_cid_tids[sub_c_to_c].index(tl1)
                idx2 = sub_cid_tids[sub_c_to_c].index(tl2)
                sim_matrix[idx1][idx2] = 0.5
                sim_matrix[idx2][idx1] = 0.5
            except: pass

        cluster_labels = AgglomerativeClustering(n_clusters=None, distance_threshold=1-dis_thrs[i], affinity='precomputed',
                                linkage='complete').fit_predict(1 - sim_matrix)
        labels = get_match(cluster_labels)
        cluster_cid_tids = get_cid_tid(labels,sub_cid_tids[sub_c_to_c])
        # print("cluster_cid_tids:",cluster_cid_tids)

        for kk in cluster_cid_tids:
            if len(kk) > 1:
                for j,v in enumerate(kk[:-1]):
                    if set([kk[j],kk[j+1]]) in tmp_list:
                        if abs(kk[j][1]-kk[j+1][1]) > 100:
                            cluster_cid_tids.remove(kk)

        sub_labels[sub_c_to_c] = cluster_cid_tids
    print("old tricklets:{}".format(len(cid_tids)))
    labels,sub_cluster = combin_cluster(sub_labels,cid_tids)

    return labels

if __name__ == '__main__':
    # Chuẩn bị tệp
    cfg.merge_from_file(f'../config/{sys.argv[1]}')
    cfg.freeze()
    scene_name = ['S06']
    scene_cluster = [[41, 42, 43, 44]]
    fea_dir = './exp/viz/test/S06/trajectory/'
    # Khởi tạo từ điến chứa quỹ đạo (key = (cid, tid))
    cid_tid_dict = dict()

    for pkl_path in os.listdir(fea_dir):
        # Lấy Camera id từ file path
        cid = int(pkl_path.split('.')[0][-3:])
        
        # lấy toàn bộ dòng trong file chứa quỹ đạo của xe cộ trong camera
        with open(opj(fea_dir, pkl_path),'rb') as f:
            lines = pickle.load(f)
        
        # Duyệt từng dòng
        for line in lines:
            # Lấy ra quỹ đạo
            tracklet = lines[line]
            # Tracklet id
            tid = tracklet['tid']
            # print(tracklet, tid)
            # Nếu trong từ điển cid_tid chưa có quỹ đạo này thì thêm vào
            if (cid, tid) not in cid_tid_dict:
                cid_tid_dict[(cid, tid)] = tracklet

    # Sắp xếp các quỹ đạo theo thứ tự từ điển và lọc chỉ giữ lại các phần tử có camera [41,42,43,44]
    # cid_tids: chứa các cặp cid-tid là key của cid_tid_dict
    cid_tids = sorted([key for key in cid_tid_dict.keys() if key[0] in scene_cluster[0]])
    # print(cid_tids)
    '''
    cid_tids = [(41, 1), (41, 2), (41, 3), (41, 4), (41, 5), (41, 6), (41, 7), (41, 9), (41, 10), 
                (41, 11), (41, 13), (41, 14), (41, 15), (41, 16), (41, 17), (41, 20), (41, 22), 
                (41, 26), (41, 27), (41, 30), (41, 39), (41, 42), (41, 44), (41, 48), (42, 2), 
                (42, 3), (42, 4), (42, 5), (42, 6), (42, 7), (42, 8), (42, 9), (42, 13), (42, 14), 
                (42, 15), (42, 16), (42, 19), (42, 21), (42, 22), (42, 23), (42, 24), (42, 25), 
                (42, 26), (42, 27), (42, 28), (42, 29), (42, 31), (42, 32), (42, 35), (42, 36), 
                (42, 39), (43, 2), (43, 3), (43, 4), (43, 5), (43, 6), (43, 7), (43, 8), (43, 9), 
                (43, 19), (43, 21), (44, 1), (44, 2), (44, 3), (44, 4), (44, 7), (44, 8), (44, 10), 
                (44, 11), (44, 13), (44, 14), (44, 18), (44, 20), (44, 21), (44, 23), (44, 24), 
                (44, 28), (44, 29), (44, 30), (44, 31), (44, 52), (44, 56), (44, 61), (44, 63), 
                (44, 71), (44, 74), (44, 76), (44, 81)]
    '''
    
    # cluster
    clu = get_labels(cfg,cid_tid_dict,cid_tids,score_thr=cfg.SCORE_THR)
    print('all_clu:', len(clu))

    new_clu = list()
    disjoint_clu = list()
    
    for c_list in clu:
        # những cluster số phần tử < 1
        if len(c_list) <= 1: 
            cam_list = [cid_tids[c][0] for c in c_list]
            disjoint_clu.append([cid_tids[c] for c in c_list])
            continue
        cam_list = [cid_tids[c][0] for c in c_list]
        new_clu.append([cid_tids[c] for c in c_list])
        
    print('disjoint clu: ', len(disjoint_clu))
    print(disjoint_clu)
    
    print('new_clu: ', len(new_clu))
    print(new_clu)
    
    all_clu = new_clu

    all_cid_tid_label = dict()
        
    # new cluster
    cid_tid_label = dict()
    for i, c_list in enumerate(all_clu):
        for c in c_list:
            cid_tid_label[c] = i + 1
            all_cid_tid_label[c] = i + 1
    pickle.dump({'cluster': cid_tid_label}, open('test_cluster.pkl', 'wb'))
    
    # disjoint cam cluster
    cid_tid_label1 = dict()
    for i, c_list in enumerate(disjoint_clu):
        for c in c_list:
            cid_tid_label1[c] = len(all_clu) +  i + 1
            all_cid_tid_label[c] = len(all_clu) +  i + 1
    pickle.dump({'cluster': cid_tid_label1}, open('disjoint_cluster.pkl', 'wb'))
    
    pickle.dump({'cluster': all_cid_tid_label}, open('all_cluster.pkl', 'wb'))
