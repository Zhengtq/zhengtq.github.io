#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import fnmatch
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import random
from glob import glob
import shutil
import time
import datetime
from math import fabs
from ztq_pylib.zfile.ztq_sm import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

LABEL_NUM = 4
RESIZE_TO = 320

label_root = './20200108_rgb.txt'
tf_root = '/media/data8/ztq/training_txt/' +label_root.split('/')[-1][:-4]  +'_'  + str(RESIZE_TO) + '_0108.txt'

filter_count = [0] * 30

img_dict = dict()
img_dict['fq_level_40'] = 0
img_dict['FaceQnet_level'] = 0
img_dict['Lp_blur_level'] = 0
img_dict['g_sharp_level'] = 0
img_dict['motionblur_level'] = 0
ztq_count = 0


def wrap_img():

	training_filename = tf_root

#          if os.path.exists(training_filename):
            #  print('File exists!')
            #  return 0


	with open(tf_root, 'w') as train_file:

		images_root,labels  = read_imgroot_lables()

                all_count = 0
                pos_count = 0
                neg_count = 0
                for ind in range(len(labels)):

                    if labels[ind] != 0:
                        labels[ind] = 1
                        pos_count += 1
                    else:
                        neg_count +=1

                    train_file.write(images_root[ind] + '####' + str(labels[ind]) + '\n')
                    all_count += 1
                print('ALL:', all_count)
                print('POS: ', pos_count)
                print('NEG: ', neg_count)

                for ind, item in enumerate(filter_count):
                    print(ind, item)


def img_quality_filter(img_q_dict, single_img_root):

    if img_dict['g_sharp_level'] < 5 and img_dict['g_sharp_level'] > 3 and img_dict['motionblur_level'] > 22 and\
    (img_dict['Lp_blur_level'] > 300 or img_dict['fq_level_40'] > 0.8 or img_dict['FaceQnet_level'] > 0.7):
#          ran_num = random.uniform(0,1)
        #  try:
            #  if ran_num<0.01:
                #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(filter_count[2]) +'.bmp')
        #  except:
            #  pass
        filter_count[0] += 1
        return True

    if img_dict['Lp_blur_level'] < 150 and img_dict['Lp_blur_level'] > 50 and img_dict['motionblur_level'] > 22 and\
    (img_dict['g_sharp_level'] > 7 or img_dict['fq_level_40'] > 0.8 or img_dict['FaceQnet_level'] > 0.7):
        filter_count[1] += 1
        return True

    if img_dict['fq_level_40'] < 0.6 and img_dict['fq_level_40'] > 0.4 and img_dict['motionblur_level'] > 22 and\
    (img_dict['Lp_blur_level'] > 300 or img_dict['g_sharp_level'] > 7 or img_dict['FaceQnet_level'] > 0.7):
        filter_count[2] += 1
        return True

    if img_dict['FaceQnet_level'] < 0.45 and img_dict['motionblur_level'] > 22 and\
    (img_dict['g_sharp_level'] > 7 or img_dict['fq_level_40'] > 0.8 or img_dict['Lp_blur_level'] > 300):
        filter_count[3] += 1
        return True

    if img_dict['motionblur_level'] < 20 and img_dict['motionblur_level'] > 5 and\
    (img_dict['g_sharp_level'] > 7 or img_dict['fq_level_40'] > 0.8 or img_dict['Lp_blur_level'] > 300):
        filter_count[4] += 1
        return True

#      if img_dict['motionblur_level'] < 20 and img_dict['motionblur_level'] > 15 and\
    #  (img_dict['g_sharp_level'] > 7 or img_dict['fq_level_40'] > 0.8 or img_dict['Lp_blur_level'] > 300):
        #  filter_count[4] += 1
        #  return True



    if img_dict['g_sharp_level'] < 5:
        filter_count[5] += 1
    if img_dict['Lp_blur_level'] < 150:
        filter_count[6] += 1
    if img_dict['fq_level_40'] < 0.6:
        filter_count[7] += 1
    if img_dict['FaceQnet_level'] < 0.45:
        filter_count[8] += 1
    if img_dict['motionblur_level'] < 20:
        filter_count[9] += 1


    if img_dict['g_sharp_level'] < 5 or img_dict['Lp_blur_level'] < 150 or img_dict['fq_level_40'] < 0.6 or img_dict['FaceQnet_level'] < 0.45 or img_dict['motionblur_level'] < 20:
        filter_count[10] += 1
        return False

    return True



def filter_img(single_img_root, label = 1):



    if '20191120-20191130_sar_phone_dump/pos/20191207' in single_img_root:
        return False


    global ztq_count
    global filter_count

###############################Filter part of Pos data#########################
    if (label > 0) and ('/3DMask_Hard/' not in single_img_root):
        pos_info_root = single_img_root[:-4] + '_score.txt'
        if not os.path.exists(pos_info_root):
            return True
        file = open(pos_info_root, 'r')
        lines = file.readlines()
        file.close()
        pos_score = float(lines[0].strip())

        pos_TH = 0.95
        if '2D_RGB_TRAIN/paper' in pos_info_root:
            pos_TH = 0.5
        if pos_score > pos_TH:
            return False
        else:
            return True

    if (label > 0) and ('/3DMask_Hard/' in single_img_root):
        pos_info_root = single_img_root[:-4] + '_score.txt'
        if not os.path.exists(pos_info_root):
            return True
        file = open(pos_info_root, 'r')
        lines = file.readlines()
        file.close()
        pos_score = float(lines[0].strip())

        pos_TH = 0.1
        if pos_score < pos_TH:
            return False
        else:
            return True


    if label > 0:
        return True


##################################################################################

    if 'ALL_NEG_DATA_4_17_NEW_croppedzzzz' in single_img_root or True:
        wh_level = 120
        fq_level_40 = 0.1
        FaceQnet_level = 0.45
        Lp_blur_level = 100
        g_sharp_level = 4.5
    else:
        wh_level = 120
        fq_level_40 = 0.1
        FaceQnet_level = 0.45
        Lp_blur_level = 200
        g_sharp_level = 4.5




##################################Open txt#######################################
    info_root = single_img_root[:-4] + '.txt'
    if not os.path.exists(info_root):
        return True

    file = open(info_root, 'r')
    lines = file.readlines()
    file.close()
    if len(lines) == 0:
        return True

##############################Get info############################################
    angle_sum = []
    origin_img_wh = []
    for ind, item in enumerate(lines):

        if 'man_sel_remove' in item:
            filter_count[20] += 1
            return False



        if 'face' in item:
            try:
                the_level = item.strip().split()[1:]
            except:
                continue
            face_rect = [int(the_level[0]), int(the_level[1]), int(the_level[2]), int(the_level[3])]

        if 'yaw' in item  or 'roll' in item or 'pitch' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            if 'roll' in item:
                the_level = 0
            angle_sum.append(the_level)

        if 'scale' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            if the_level < 0.2:
                filter_count[21] += 1
                return False

        if 'fq_level_40' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue

            if '7_12_5_floor' in single_img_root:
                the_level = 1
            img_dict['fq_level_40'] = the_level


        #    if the_level < fq_level_40:
#                  ran_num = random.uniform(0,1)
                #  try:
                    #  if ran_num<0.1:
                        #  if face_rect[2] - face_rect[0] > 120:
                            #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(the_level) +"_" + str(face_rect[2] - face_rect[0])  + '.jpg')
                #  except:
                    #  pass

        if 'FaceQnet' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            img_dict['FaceQnet_level'] = the_level
         #   if the_level < FaceQnet_level:
  #                ran_num = random.uniform(0,1)
                #  try:
                    #  if ran_num<0.5:
                        #  if face_rect[2] - face_rect[0] > 120:
                            #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(the_level) +"_" + str(face_rect[2] - face_rect[0])  + '.jpg')
                #  except:
                    #  pass

        if 'Lp_blur' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            img_dict['Lp_blur_level'] = the_level
        #    if the_level < Lp_blur_level:
#                  ran_num = random.uniform(0,1)
                #  try:
                    #  if ran_num<0.001:
                        #  if face_rect[2] - face_rect[0] > 120:
                            #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(the_level) +"_" + str(face_rect[2] - face_rect[0])  + '.jpg')
                #  except:
                    #  pass

        if 'g_sharp' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            img_dict['g_sharp_level'] = the_level
       #     if the_level < g_sharp_level:
#                  ran_num = random.uniform(0,1)
                #  try:
                    #  if ran_num<0.1:
                        #  if face_rect[2] - face_rect[0] > 120:
                            #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(the_level) +"_" + str(face_rect[2] - face_rect[0])  + '.jpg')
                #  except:
                    #  pass


        if 'MotionBlurDet' in item:
            try:
                the_level = float(item.strip().split(':')[1])
            except:
                continue
            img_dict['motionblur_level'] = the_level
#              if the_level < 18:
                #  filter_count[16] += 1
                #  ztq_copy(single_img_root, '/home/ztq/show/' + str(the_level) + '.bmp')
                #  ztq_count += 1
                #  return False


        if 'width_true' in item or 'height_true' in item:
            if len(origin_img_wh) == 2:
                continue
            try:
                the_level = float(item.strip().split(':')[1])
                origin_img_wh.append(the_level)
            except:
                continue

#          if 'w_h' in item:
            #  if len(origin_img_wh) == 2:
                #  continue
            #  try:
                #  the_level1 = float(item.strip().split(':')[1].split(',')[0])
                #  the_level2 = float(item.strip().split(':')[1].split(',')[1])
                #  origin_img_wh.append(the_level1)
                #  origin_img_wh.append(the_level2)
            #  except:
                #  continue

    img_quality_flag = img_quality_filter(img_dict, single_img_root)
    if not img_quality_flag:
        return False


####################################filter angle##################
    if len(angle_sum) > 0 and(
        ((angle_sum[0]>30 or angle_sum[1]<-30) and
        (angle_sum[1]>30 or angle_sum[1]<-30) and
        (angle_sum[2]>30 or angle_sum[2]<-30)) or
        angle_sum[0] > 50 or angle_sum[0] < -50 or
        angle_sum[1] > 50 or angle_sum[1] < -50 or
        angle_sum[2] > 50 or angle_sum[2] < -50):
        filter_count[22] += 1
        return False

#########################filter area#########################
    if 'face' not in lines[0]:
        return True
    rect_line = face_rect
    x1 = float(rect_line[0])
    y1 = float(rect_line[1])
    x2 = float(rect_line[2])
    y2 = float(rect_line[3])
    face_w = x2-x1
    face_h = y2-y1

    if face_w < wh_level or face_h< wh_level:
   #     ran_num = random.uniform(0,1)
#          if ran_num<0.001:
            #  ztq_copy(single_img_root, '/home/ztq/show/'+ str(the_level) + '.jpg')
        filter_count[23] += 1
        return False
#####################################filter scale###################
    try:
        if origin_img_wh == []:
            origin_img_wh = [640, 480]
        face_scale = max(fabs(rect_line[0] - rect_line[2]) , fabs(rect_line[1] - rect_line[3])) / min(origin_img_wh[0] , origin_img_wh[1])
        if face_scale < 0.2:
            filter_count[24] += 1
            return False
    except:
          pass


    return True



def generate_img_roots(file_root, search_heard = '*.jpg', label = 1, use_filter = True):
    imgs_root = []

    all_count = 0
    valid_count = 0
    for dirpath, dirnames, files in os.walk(file_root):
        for f in fnmatch.filter(files, search_heard):

            single_path = os.path.join(dirpath, f)

            if not use_filter:
                valid_img = True
            else:
                valid_img = filter_img(single_path, label = label)

            if valid_img:
           #     if label == 0:
#                      ran_num = random.uniform(0,1)
                    #  if ran_num<0.001:
                        #  ztq_copy(single_path, '/home/ztq/show/' + str(valid_count) +str(ran_num)[:3] + '.jpg')
                imgs_root.append(single_path)
                valid_count += 1

            all_count += 1

    if all_count > 3000:
        print(all_count, valid_count)

    return imgs_root


def generate_equal_img_roots(file_root, keep_num = 0,search_heard = ['.bmp'], label = 1):



    file_roots_tmp = generate_img_roots(file_root, search_heard=search_heard, label = label, use_filter = False)
    subfolders = []
    for ind, item in enumerate(file_roots_tmp):
        single_folder = os.path.dirname(item)
        if single_folder in subfolders:
            continue
        subfolders.append(single_folder)


    all_img_root = []
    for folder in subfolders:
        imgs = generate_img_roots(folder, search_heard=search_heard, label = label)
        img_num = len(imgs)
        if img_num <= keep_num:
            all_img_root += imgs
        else:
            random.shuffle(imgs)
            all_img_root += imgs[:keep_num]

    return all_img_root


def ran_1080_640(img_roots):


    new_img_roots = img_roots
    for ind,item in enumerate(img_roots):

        other_root = item.replace('crop_train_after_640_resized320bmp', 'crop_train_after_1080_resized320bmp')
        if 'crop_train_after_1080_resized320bmp' not in other_root:
            break
        if os.path.exists(other_root):
            if random.uniform(0, 1) < 0.5:
                item = other_root

        new_img_roots[ind] = item


    return new_img_roots


def read_imgroot_lables():

    with open(label_root, 'r') as label_file:

	img_label_pos = []
        img_label_neg = []
        label_num = [0] * LABEL_NUM
	for line in label_file:

                if line == '':
                    break

                try:
                     label, img_folder = line.split()[:2]
                except:
                    print('FALSE>>', line)
                    continue

                if len(line.split()) == 3 and float(line.split()[2]) > 1.01:
                    imgs_root = generate_equal_img_roots(img_folder, keep_num = int(line.split()[2]), search_heard='*.bmp', label = int(label))
                else:
                    imgs_root = generate_img_roots(img_folder, '*.bmp', label = int(label))

                imgs_root = ran_1080_640(imgs_root)

                if len(line.split()) == 3 and float(line.split()[2]) <= 1:
                    sample_rate = float(line.split()[2])
                    random.shuffle(imgs_root)
                    imgs_root = imgs_root[:int(len(imgs_root) * sample_rate)]

                label_num[int(label)] += len(imgs_root)
                print (img_folder, len(imgs_root))

                random.shuffle(imgs_root)
		for ind ,root in enumerate(imgs_root):
                    if int(label) == 0:
                        img_label_neg.append([root, int(label)])
                    else:
                        img_label_pos.append([root, int(label)])


        random.shuffle(img_label_neg)
        random.shuffle(img_label_pos)


        print(len(img_label_neg))
        print(len(img_label_pos))

        img_label = []
        pos_ind = 0
        neg_ind = 0
        for ind in range(max(len(img_label_neg), len(img_label_pos))):

            if pos_ind == len(img_label_pos):
                pos_ind = 0
            if neg_ind == len(img_label_neg):
                neg_ind = 0

            img_label.append(img_label_pos[pos_ind])
            pos_ind +=1
            img_label.append(img_label_neg[neg_ind])
            neg_ind +=1

        PNUM = 0
        NNUM = 0
        for label_ind in range(LABEL_NUM):
            print(label_ind, ':',label_num[label_ind])
            if label_ind == 0:
                NNUM += label_num[label_ind]
            else:
                PNUM += label_num[label_ind]
        print('ALL_POS:', PNUM, 'ALL_NEG:', NNUM)


	img_all = np.array([x[0] for x in img_label])
	label_all = np.array([x[1] for x in img_label])


	return img_all, label_all


if __name__ == '__main__':
    wrap_img()
