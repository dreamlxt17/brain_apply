# coding=utf-8

import cv2
import mahotas
from glcm import GLCM
import numpy as np


def comput_hog(roi, shape=208):
    '''

    :param roi: 一个人的所有roi slice
    :return: 每张slice 36个 histogram 特征
    '''
    winSize = (shape, shape)
    blockSize = (shape, shape)
    blockStride = (shape/2, shape/2)
    cellSize = (shape/2, shape/2)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradients = True
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)
    feature_list = []
    for slice in roi:
        slice = slice.astype(np.uint8)
        descriptor = hog.compute(slice)
        descriptor = np.squeeze(descriptor)
        feature_list.append(descriptor)
    return feature_list

def comput_haralick(roi, shape):
    '''

    :param roi: 一个人的所有roi切片
    :return: 每张slice 13个harlick特征
    '''
    feature_list = []
    for slice in roi:
        slice = slice.astype(np.uint8)
        descriptor = mahotas.features.haralick(slice).mean(0)
        feature_list.append(descriptor)
    return feature_list

def comput_glcm(roi, shape):
    """
    :param roi: 一个人的所有roi切片
    :param shape:
    :return: 每张切片的 99 个特征
    """
    feature_list = []
    for slice in roi:
        feature_list.append(GLCM(np.array(slice)).get_features())
    return feature_list

def comput_shape(roi, shape):
    '''
    计算形状特征
    :param roi: 计算形状特征
    :return: 每张slice 7个特征
    '''
    feature_list = []
    for slice in roi:
        slice = slice.astype(np.uint8)
        # 纵横比  Aspect Ratio = (width / height)
        x, y, w, h = cv2.boundingRect(slice)
        rect_area = w * h
        # 部分数据全黑, 故分母可能为0, 当分母为0时，令aspect_ratio, extent 为0
        if h > 0:
            aspect_ratio = float(w) / h
            image_roi = slice
            img, contours, hierarchy = cv2.findContours(image_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            extent = float(area) / rect_area
            equi_diameter = np.sqrt(4 * area / np.pi)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                solidity = 0
            else:
                solidity = float(area) / hull_area
            mask = np.zeros(slice.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_val = np.mean(np.asarray(cv2.mean(slice, mask)))
            M = cv2.moments(cnt)
            m20 = M['m20']
            m02 = M['m02']
            m11 = M['m11']
            if (m20 + m02) == 0:
                eccentricity = 0
            else:
                eccentricity = (m20 - m02 * m02 + 4 * m11 * m11) * 1.0 / ((m20 + m02) * (m20 + m02))
            perimeter = cv2.arcLength(cnt, True)
            hullperimeter = cv2.arcLength(hull, True)
            if hullperimeter == 0:
                roughness = 0
            else:
                roughness = perimeter / hullperimeter
            # orientation is the angle at which object is directed.
            # (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        else:
            aspect_ratio = 0
            extent = 0
            equi_diameter = 0
            solidity = 0
            mean_val = 0
            eccentricity = 0
            roughness = 0

        feature_list.append([aspect_ratio, extent, equi_diameter, solidity, mean_val, eccentricity, roughness])
    return feature_list