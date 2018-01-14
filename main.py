# coding=utf-8
import dicom
import numpy as np
import torch
from glob import glob
from scipy.ndimage import zoom
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.externals import joblib
from features import comput_shape, comput_hog, comput_haralick, comput_glcm
import pickle

parser = argparse.ArgumentParser(description='Brain Cancer Classify')

parser.add_argument('--classifymodel',
                    default='/all/DATA_PROCEING/classify_model/',
                    help='model for classifying, including KNN, GB, RF')

parser.add_argument('--unet208model',
                    default='/home/didia/Didia/examples/unet/brain/save_model_1/1999.ckpt',  # 1099
                    help='model for shape-208')

parser.add_argument('--inputdir',
                    default='/all/DATA_PROCEING/total_original_data/0/edemaYAN_FEN_FA',
                    help='input dcm file for one person')

parser.add_argument('--outputdir',
                    default='/all/DATA_PROCEING/result',
                    help='results of classify or sementation')

parser.add_argument('--clftype',
                    default='RF',
                    help='chose a classifier')

def load_model():
    '''
    获取u-net model,
    :param para: =1 时为208的model; =2时为512的model
    :return: pre-train好的unet-model
    '''
    from unet import UnetSegment as Unet
    unet = Unet(1, 2)
    print 'loading model :', args.unet208model
    checkpoint = torch.load(args.unet208model)
    unet.load_state_dict(checkpoint['state_dict'])
    #unet.cuda()
    return unet

def get_roi(sample, unet, shape=208, save_dir=None):
    '''
    获取一个人每张切片的预测roi
    :param sample: 一个人的dcm文件夹路径,如'/all/DATA_PROCEING/total_original_data/0/edemaTANG_XIAO_YUN/'
    :return: 所有含roi的slice
    '''
    len_sample = len(glob(sample+'/*.dcm'))
    name = sample.split('/')[-1]
    roi = []
    for i in range(len_sample):
        # print i
        dcm = dicom.read_file(sample+'/'+str(i+1)+'.dcm').pixel_array
        dcm = zoom(dcm, shape*1.0/dcm.shape[0])
        data = torch.from_numpy(np.array(dcm).reshape(1, 1, shape, shape).astype(np.float))
        #batch_data = Variable(data.type(torch.FloatTensor).cuda()) 
        batch_data = Variable(data.type(torch.FloatTensor))
        outputs, _ = unet(batch_data)
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.cpu().numpy().reshape(shape, shape)
        if np.sum(prediction)>100 and i>2:
            # 筛选包含roi区域的slice
            roi.append(prediction*dcm)
            if save_dir:
                if not os.path.exists(save_dir+'/'+name):
                    os.makedirs(save_dir+'/'+name)
                np.save(save_dir+'/'+name+'/{}.npy'.format(i), prediction*dcm)
                plt.imshow(prediction*dcm, label='Predicted', cmap='gray')
                plt.savefig(save_dir+'/'+name+'/{}.png'.format(i))
    return roi

def get_feature(roi, feature_type=['glcm', 'hog', 'haralick', 'shape'], shape=208):
    '''
    获取一个人每张roi的对应feature
    :param path:  feature存储路径
    :param feature_type: feature 类型
    :return: 一个人的一组feature， 以slices的形式返回
    '''
    feature_dict = {'glcm':comput_glcm, 'haralick':comput_haralick,
                    'hog':comput_hog, 'shape':comput_shape}
    feature_list = list(0 for i in range(len(feature_type)))
    for i, feature in enumerate(feature_type):
        computer = feature_dict[feature]
        f = np.array(computer(roi, shape))
        feature_list[i] = f
    features = np.hstack(feature_list)
    print 'shape of feature:', features.shape
    return features

def classfiy(features, clf_type='GB', feature_type=['glcm', 'hog', 'haralick', 'shape']):
    '''
    预测每张slice是转移瘤还是胶质瘤， 预测每个人是转移瘤还是胶质瘤
    :param features:一个人的所有roi切片对应的feature
    :param clf: 分类器类型， 候选参数： GB, RF, KNN
    :return 每张slice分类准确率，最后投票准确率
    '''
    scaler = joblib.load(args.classifymodel + 'scaler.m')
    features = scaler.transform(features)

    shape_dict = {'glcm':11, 'haralick':13,
                    'hog':36, 'shape':7}
    if clf_type == 'RF':
        # clf = RandomForestClassifier(n_estimators=40, criterion='entropy',max_depth=12)  # min_samples_split=192)
        clf = joblib.load(args.classifymodel + 'RF_67.m')
    elif clf_type == 'KNN':
        # clf = neighbors.KNeighborsClassifier(leaf_size=20)
        clf = joblib.load(args.classifymodel + 'KNN_67.m')
    else:
        clf_type = 'GB'
        # clf = GradientBoostingClassifier(n_estimators=180, max_depth=4, min_samples_split=22)
        clf = joblib.load(args.classifymodel + 'GB_67.m')

    prediction = clf.predict(features)
    print prediction
    pred_1 = np.sum(prediction)*1.0/len(prediction)
    pred_0 = 1 - pred_1
    if pred_1 >= pred_0:
        label = 1
    else:
        label = 0
    print pred_0, pred_1, '\n', label
    return prediction, label

def main(dcms_path='', feature_type=['glcm', 'hog', 'haralick', 'shape'], clf_type='RF', save_dir=None):
    '''
    输入一个人的dcm文件夹路径，得到其胶质瘤/转移瘤分类预测结果
    :param name_path: 名字路径
    :param feature_type: 选取的feature类型
    :param para: 选哪种model
    :param clf_type: 分类器
    :return: 不同分类器给出的分类结果
    '''
    # name_path = data_dir + 'edemaTANG_XIAO_YUN/'
    name = dcms_path.split('/')[-1] # 获取病人名称
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    unet = load_model()
    shape=208
    roi = get_roi(dcms_path, unet, shape, save_dir)
    features = get_feature(roi, feature_type=feature_type)
    prediction, label = classfiy(features, clf_type=clf_type)

    if save_dir:
        pickle.dump([prediction, label], open(save_dir+ '/' + name + '/'+ name +'.pkl', 'w'))
    return prediction, label

if __name__ == '__main__':

    args = parser.parse_args()
    main(dcms_path=args.inputdir, save_dir = args.outputdir, clf_type=args.clftype)







