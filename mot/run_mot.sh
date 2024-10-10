#!/bin/bash
# Chạy tiền xử lý
# Done: 6/10/2024
cd tool
python pre_process.py

# Tạo thư mục build chứa kết quả biên dịch
# Done
cd ..
if [ ! -d "./build" ];then
    mkdir build
else
    rm -rf ./build/*
fi

# Vào thư mục build, sử dụng 4 luồng song song, chạy thư mục city_tracker
cd build && cmake .. && make -j4
./city_tracker

# Chạy post processing
cd ../tool
python post_precess.py

seqs=(c041 c042 c043 c044)
# seqs=(c042)

# Hàm lưu kết quả của từng camera độc lập
TrackOneSeq(){
    seq=$1
    config=$2
    echo save_sot $seq with ${config}
    # Lưu lại kết quả
    python save_mot.py ${seq} pp ${config}
}

# Chạy duyệt qua chuỗi camera và lwuu kqua từng camera
for seq in ${seqs[@]}
do 
    # Dấu & --> chạy song song nhiều tiến trình
    TrackOneSeq ${seq} $1 &
done
wait
