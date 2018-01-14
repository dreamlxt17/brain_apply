# An end2end brain tumor segmentation and classify system

# 安装依赖包
pip install dicom
pip install mahotas

# 执行mian.py
# 共有5个参数
# --classifymodel: 分类器模型路径
# --unet208model: segmentation模型路径
# --inputdir: 病人dcm文件夹路径
# --outputdir: 预测结果存储路径，包括预测ROI及slice分类结果
# --clftype: 选择分类器类型 ['RF', 'GB', 'KNN']

python main.py --classifymodel /all/DATA_PROCEING/classify_model/ --inputdir /all/DATA_PROCEING/total_original_data/0/edemaYAN_FEN_FA --unet208model /home/didia/Didia/examples/unet/brain/save_model_1/1979.ckpt --outputdir /all/DATA_PROCEING/result --clftype 'RF'
