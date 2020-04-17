from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')
sys.path.append('./shufflenet_v2_1_depth/')

import tf_data_flow as td
import tensorflow as tf
import numpy as np
import random
import shufflenet_v2 as face_train
import math
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import slim
import cv2
from glob import glob
import os
import datetime
from tensorflow.python.client import timeline
import operator
from ztq_pylib.ztf_git.training_config_cls import *

from noise_generate import noise_generator

np.set_printoptions(precision=3, suppress=True)


RESIZE_TYPE = 10
IMG_NORM = False
RESTORE = True
USE_CENTER_LOSS = False
ALL_VAL_NUM = 10800
split_time = 108
SPLIT_NUM = ALL_VAL_NUM // split_time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
GPU_NUM_ID = [0,1,2,3]
#GPU_NUM_ID = [0,1,2,3]
VAL_GPU_ID = 3
########## web_face setting ############320#
EPOC_NUM = 50
BATCH_SIZE = 128
BATCH_SIZE_SEP = BATCH_SIZE//len(GPU_NUM_ID)
TEST_SIZE = 320
RAN_CROP_SIZE = 280
SAMPLE_NUM = 2594870
ORIGINAL_SIZE = [320, 320, 3]
CHANNELS = 3
LABEL_NUM = 1
validate_txt_root = './20190918_val.txt' 
log_dir = '/media/data7/ztq/log_now/202002_12_advnoise_1/'


LOSS_FUN = COM_LOSS()
ACC_FUN = COM_ACC()


def validate_frequence(epoc):

    if epoc < EPOC_NUM//5:
        return 1000
    if epoc >=EPOC_NUM//5 and epoc < EPOC_NUM // 4:
        return 500
    if epoc >= EPOC_NUM//4:
        return 250

    return 50

def make_rm_log_dir():

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    files = glob(log_dir + '/*')

    for rmfile in files:
        if rmfile.split('.')[-1] == 'py':
            continue
        os.remove(rmfile)

def load_validate_img_data():

    img_roots = []
    with open(validate_txt_root, 'r') as validate_file:
        for item in validate_file:
            img_roots.append(item.strip().split('#@#'))

    random.shuffle(img_roots)
    validate_img = []
    validate_label = []
    validate_img_root = []
    read_count = 0
    for root in img_roots:
        if read_count == ALL_VAL_NUM:
            break
        the_img = cv2.imread(root[0])
        if the_img is None:
            continue

#          if RESIZE_TYPE == 0:
            #  the_img = cv2.resize(the_img, (TEST_SIZE, TEST_SIZE))
        #  if RESIZE_TYPE == 1:
            #  the_img = cv2.resize(the_img, (TEST_SIZE, TEST_SIZE), interpolation=cv2.INTER_NEAREST)

        validate_img_root.append(root[0])
        validate_img.append(the_img)
        validate_label.append(int(root[1]))
        read_count += 1
        if read_count % 1000 == 0:
          print (read_count)

    if len(validate_img) != ALL_VAL_NUM:
        validate_img.extend(validate_img[:ALL_VAL_NUM-len(validate_img)])
        validate_label.extend(validate_label[:ALL_VAL_NUM-len(validate_label)])
        validate_img_root.extend(validate_img_root[:ALL_VAL_NUM-len(validate_img)])

    validate_img = np.array(validate_img).astype(np.uint8)
    validate_label = np.array(validate_label).astype(np.int64)

    return validate_img, validate_label, validate_img_root

def generate_validate_input_output():

    with tf.device('/gpu:%d' % VAL_GPU_ID):
        with tf.name_scope('validate_input'):
            input_imgs = tf.placeholder(tf.uint8, shape = (SPLIT_NUM, TEST_SIZE, TEST_SIZE, CHANNELS), name = 'imgs')
            transfer_input_imgs, _ = td._resize_crop_img(input_imgs, TEST_SIZE, TEST_SIZE, process_type = 'validate', img_norm = IMG_NORM, resize_type = RESIZE_TYPE)

        with tf.name_scope('validate_output'):
            val_logits, out_data = face_train.inference(transfer_input_imgs, num_classes = LABEL_NUM, is_training = False, reuse = True)

    return input_imgs, val_logits


def produce_lr(now_step, epoc_step):

    base_lr = 0.1
    if now_step < epoc_step:
        return 0.01
    else:
        now_lr = base_lr * math.pow(0.5, int((now_step - epoc_step) / (epoc_step * 3)))
        return now_lr




def train_cls():
 #   with tf.Graph().as_default():


            ######################
            optimizer_G = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon=0.1)
            optimizer_ord = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon=0.1)
            #######################


            tf.logging.set_verbosity(tf.logging.INFO)

            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
       #     lr_in_use = tf.train.exponential_decay(0.1, global_step, SAMPLE_NUM/BATCH_SIZE * EPOC_NUM, 0.003, staircase=False)
        #    lr_in_use = tf.train.exponential_decay(0.1, global_step, SAMPLE_NUM/BATCH_SIZE * 3, 0.5, staircase=True)
        #    lr_in_use = tf.train.polynomial_decay(0.1, global_step, SAMPLE_NUM/BATCH_SIZE * EPOC_NUM, 0.001, power=3, cycle=False)

            lr_in_use = tf.placeholder(tf.float32, shape=[])
            lr_in_use = tf.maximum(lr_in_use, 0.0001)
            optimizer = tf.train.MomentumOptimizer(lr_in_use, 0.9)
        #    optimizer = tf.train.AdamOptimizer(learning_rate = lr_in_use, epsilon=0.1)
         #   optimizer = tf.contrib.opt.AdamWOptimizer(weight_decay = 0.000000001,learning_rate = lr_in_use, epsilon = 0.1)
         #   optimizer = tf.train.RMSPropOptimizer(learning_rate = lr_in_use, decay=0.9, momentum=0.9, epsilon=1.0)

            ##############
            tower_grads_G = []
            tower_grads_ord = []
            #############

            tower_grads = []
            img_roots_txt = '/media/data8/ztq/training_txt/20200108_rgb_320_0108.txt'
            input_queue, enqueue_op, img_root_list, label_list, index_dequeue_op,\
            image_paths_placeholder, labels_placeholder =td._generate_list(img_roots_txt, BATCH_SIZE, EPOC_NUM)
            input_size = input_queue.size()
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(len(GPU_NUM_ID)):

                    with tf.device('/gpu:%d' % GPU_NUM_ID[i]):
                        with tf.name_scope('%s_%d' % ('cnn_mg', i)) as scope:

                            images, images_depth, labels = td._load_batch_filename(input_queue, ORIGINAL_SIZE, CHANNELS, BATCH_SIZE_SEP,SAMPLE_NUM,
                                RAN_CROP_SIZE, RAN_CROP_SIZE, img_norm = IMG_NORM, resize_type = RESIZE_TYPE)


                            ############################
                          #  ran_sel = tf.ones([BATCH_SIZE_SEP, 1, 1, 1], tf.int32)
                        #    ran_sel = tf.zeros([BATCH_SIZE_SEP, 1, 1, 1], tf.int32)
                            ran_sel = tf.random_uniform((BATCH_SIZE_SEP, 1, 1, 1), minval=0,maxval=2,dtype=tf.int32)
                            ran_sel = tf.cast(ran_sel, tf.float32)
                            g_noise_input = tf.random_normal(shape=(BATCH_SIZE_SEP, 280, 280, 3), mean=0.0, stddev=0.5, dtype=tf.float32)
                        #    g_noise_input = g_noise_input * ran_sel
                            out_noise = noise_generator(g_noise_input)
                       #     ord_loss = tf.sqrt(tf.pow(tf.norm(out_noise, ord=2)/10000 - 1, 2))

  #                            tmp_noise = tf.reshape(out_noise, (BATCH_SIZE_SEP, 280*280*3))
                            #  ztqtmp = tf.norm(tmp_noise, ord=2, axis=-1)

                            tmp_noise = tf.reshape(out_noise, (BATCH_SIZE_SEP, 280*280*3))
                            ord_loss = tf.norm(tmp_noise, ord=2, axis=-1)
                            ord_loss = tf.sqrt(tf.pow(ord_loss - 2000, 2))/200
                            ooo = ord_loss
                            ord_loss = tf.reduce_mean(ord_loss)
                            
                            out_noise = out_noise * ran_sel
                        #    out_noise = out_noise * 100
                            black_mask = tf.not_equal(images, 0)
                            black_mask = tf.cast(black_mask, tf.float32)
                            noise_image = images + out_noise
                         #   noise_image = images + 100
                            noise_image = tf.multiply(noise_image,black_mask)
                            noise_image = tf.clip_by_value(noise_image, 0.0, 255.0)
                            fake_logits, fake_out_data = face_train.inference(noise_image,  num_classes=LABEL_NUM)

                            reverse_label = tf.cast((1-labels), tf.float32)
                         #   reverse_label = tf.cast((labels), tf.float32)
                            loss_softmax_sep_fake = LOSS_FUN._focal_loss_4(reverse_label, fake_logits, ran_sel)
                           
                            tf.add_to_collection('train_loss_G', loss_softmax_sep_fake)
                            tf.add_to_collection('train_loss_ord', ord_loss)

                            G_variable = [var for var in tf.trainable_variables() if 'advnoise' in var.name]
                            grads_G = optimizer_G.compute_gradients(loss_softmax_sep_fake, var_list=G_variable)
                            tower_grads_G.append(grads_G)
                            
                            grads_ord = optimizer_ord.compute_gradients(ord_loss, var_list=G_variable)
                            tower_grads_ord.append(grads_ord)
                            ############################
                            
                            logits = fake_logits
                            out_data = fake_out_data
                        #    logits, out_data = face_train.inference(images,  num_classes=LABEL_NUM)

                            if not USE_CENTER_LOSS:
                                out_data['features'] = None

                            loss_softmax_sep = LOSS_FUN._focal_loss_3(labels, logits)
                         #   out_data['depth'] = tf.nn.sigmoid(out_data['depth'])
                            loss_depth_sep = LOSS_FUN._get_depth_loss(images_depth, out_data['depth'])
                            loss_depth_sep = loss_depth_sep * 0.3
                            loss_regulation = LOSS_FUN._get_regulation_loss()
                            accuracy_sep, _ = ACC_FUN._get_train_acc(logits, labels, label_num = 1)
                            loss_total_sep = loss_softmax_sep + loss_depth_sep + loss_regulation

                            tf.add_to_collection('train_softmax_loss', loss_softmax_sep)
                            tf.add_to_collection('train_depth_loss', loss_depth_sep)
                            tf.add_to_collection('train_acc', accuracy_sep[0])
                            tf.add_to_collection('train_total_loss', loss_total_sep)
                            tf.add_to_collection('train_per_class_acc', accuracy_sep[1:])



                            tf.get_variable_scope().reuse_variables()
                            D_variable = [var for var in tf.trainable_variables() if 'ShuffleNetV2' in var.name]
                            grads = optimizer.compute_gradients(loss_total_sep,var_list=D_variable)
                            tower_grads.append(grads)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads = average_gradients(tower_grads)
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
                grads_G = average_gradients(tower_grads_G)
                train_op_G = optimizer_G.apply_gradients(grads_G, global_step=global_step)
                grads_ord = average_gradients(tower_grads_ord)
                train_op_ord = optimizer_ord.apply_gradients(grads_ord, global_step=global_step)


            train_accuracy = tf.reduce_mean(tf.stack(tf.get_collection('train_acc')))
            train_total_loss = tf.reduce_mean(tf.stack(tf.get_collection('train_total_loss')))
            train_softmax_loss = tf.reduce_mean(tf.stack(tf.get_collection('train_softmax_loss')))
            train_depth_loss = tf.reduce_mean(tf.stack(tf.get_collection('train_depth_loss')))
            train_per_class_acc = tf.reduce_mean(tf.stack(tf.get_collection('train_per_class_acc')),axis=0)
            train_depth_loss_pos = tf.reduce_mean(tf.stack(tf.get_collection('train_depth_loss_pos')))
            train_depth_loss_neg = tf.reduce_mean(tf.stack(tf.get_collection('train_depth_loss_neg')))
            train_loss_G = tf.reduce_mean(tf.stack(tf.get_collection('train_loss_G')))
            train_loss_ord = tf.reduce_mean(tf.stack(tf.get_collection('train_loss_ord')))



            validate_img, validate_label, validate_img_root = load_validate_img_data()
            img_placeholer, validate_logits = generate_validate_input_output()
            val_input_logits_pl = tf.placeholder(tf.float32, shape = (None), name = 'logits_ztq')
            val_input_labels_pl = tf.placeholder(tf.int64, shape = (None), name = 'labels_ztq')
            validate_accuracy, val_predicted_lables = ACC_FUN._get_train_acc(val_input_logits_pl, val_input_labels_pl, label_num= 1, name="metrics_test")
            validate_loss = LOSS_FUN._focal_loss_2(val_input_labels_pl, val_input_logits_pl)
            metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics_test")
            metrics_init_op = tf.variables_initializer(var_list=metrics_vars, name='validation_metrics_init')

            tf.summary.scalar('train/learning_rate', lr_in_use)
            tf.summary.scalar('train/total_loss', train_total_loss)
            tf.summary.scalar('train/softmax_loss', train_softmax_loss)
            tf.summary.scalar('train/depth_loss', train_depth_loss)
            tf.summary.scalar('train/regulizer_loss', loss_regulation)
            tf.summary.scalar('train/accuracy', train_accuracy)
            tf.summary.scalar('train/depth_loss_pos', train_depth_loss_pos)
            tf.summary.scalar('train/depth_loss_neg', train_depth_loss_neg)
            tf.summary.scalar('validate/accuracy', validate_accuracy[0])
            tf.summary.scalar('validate/softmax_loss', validate_loss)
            tf.summary.image('depth_img_gt', images_depth, max_outputs = 10, family = 'depth_gt')
            tf.summary.image('depth_img_pd', out_data['depth'] * 255, max_outputs = 10, family = 'depth_pd')

            merged_train = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'train') +
                   tf.get_collection(tf.GraphKeys.SUMMARIES,'depth_'))
            merged_val = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,'validate'))

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement=True
            with tf.Session(config = config) as sess:
                    make_rm_log_dir()
                    writer = tf.summary.FileWriter(log_dir,sess.graph)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())

                    if RESTORE:
                            restore_var = []
                            for item in tf.global_variables():
                                 if 'advnoise' in item.op.name or 'beta1_power' in item.op.name\
                                         or 'beta2_power' in item.op.name:
                                        continue
                                 restore_var.append(item)
                            restore_variable = restore_var
                         #   restore_variable = tf.global_variables()
                            reload_saver = tf.train.Saver(restore_variable, max_to_keep=10000)
                         #   checkpoint_path = '../../pretrained_model/shufflenet_v2_imagenet/run01/model.ckpt-1661328'
                       #     checkpoint_path = './shufflenet_v2_imagenet/run00/model.ckpt-1331064'
                            checkpoint_path = '/media/data8/ztq/log_backup/v1.1.0.14/202001_09_model/0.980_auc:0.9981_loss:0.016_presion:0.976_recall:0.986_model.kept-343500'
                            reload_saver.restore(sess, checkpoint_path)
                    saver = tf.train.Saver(tf.global_variables(),max_to_keep=10000)

                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

                    start_time = datetime.datetime.now()
                    epoc_step = int(SAMPLE_NUM/(BATCH_SIZE))
                    c_count = 0
                    g_step = 0
                    ord_loss_out = 1000000
                    for loop in range(10000000):

                            input_size_out = sess.run(input_size)
                            if input_size_out < BATCH_SIZE * 2:
                                index_epoch = sess.run(index_dequeue_op)
                                label_epoch = np.array(label_list)[index_epoch]
                                image_epoch = np.array(img_root_list)[index_epoch]
                                labels_array = np.expand_dims(np.array(label_epoch),1)
                                image_paths_array = np.expand_dims(np.array(image_epoch),1)
                                sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})


                            if loop % 1001 ==0:
                                sess.run(tf.variables_initializer(G_variable))
                                ord_loss_out = 1000000


                            if loop % 50 == 0:
                                ord_loss_out_notrain, ztq_out_noise = sess.run([ord_loss, out_noise])
                                print(ztq_out_noise[0,:,:,0])
                           #     print(ord_loss_out_notrain, ztq_out_noise[0,:,:,0])

                            if ord_loss_out > 0.1:
                                _, ord_loss_out = sess.run([train_op_ord, ord_loss])
                                print('step:', loop, 'ord_loss:',ord_loss_out)
                                continue
                           

                            _, G_loss_out = sess.run([train_op_G, train_loss_G])
                            print('---------------------------')
                            print('step:',loop, 'G_loss:', G_loss_out)

                            now_lr = produce_lr(g_step, epoc_step)
                            if loop != 0:
                                _, g_step= sess.run([train_op, global_step], feed_dict={lr_in_use: now_lr})
                            else:
                                g_step = sess.run(global_step, feed_dict={lr_in_use:now_lr})

                            g_step = int(g_step)
                            epoc = g_step / float(epoc_step)
                            if epoc >= EPOC_NUM:
                                break

                            print(g_step)
                            if loop % 50 == 0 or loop == 0:
                                    end_time = datetime.datetime.now()
                                    time_interval = (end_time - start_time).seconds
                                    start_time = datetime.datetime.now()
                                    epoc_time = time_interval * epoc_step / 3600.0/ 100


                                    per_class_acc_out,loss,softmax_loss_out, depth_loss_out, \
                                    regulizer_loss_out, train_ac, summary_train=sess.run([train_per_class_acc,train_total_loss, \
                                    train_softmax_loss, train_depth_loss,loss_regulation,train_accuracy, merged_train], feed_dict={lr_in_use:now_lr})
                                    writer.add_summary(summary_train,g_step)

                                    print ('epoc : %.3f , loss : %.2f, softmax_loss : %.4f, depth_loss : %.4f, regulizer_loss : %.4f, train_acc : %.3f, epoc_time : %.2fh' \
                                            %(epoc,  loss, softmax_loss_out, depth_loss_out,regulizer_loss_out,train_ac, epoc_time),\
                                            per_class_acc_out)



                            if g_step  % 30000 == 0  or g_step == 20:
                                save_model(saver, sess, log_dir, g_step)


                            val_step = validate_frequence(epoc)
                            if loop % val_step == 0 or loop == 0:

                                all_val_logits = []
                                all_val_labels = []
                                for s in range(split_time):
                                    batch_val_img_root = validate_img_root[s*SPLIT_NUM:(s+1)*SPLIT_NUM]
                                    batch_val_img = validate_img[s*SPLIT_NUM:(s+1)*SPLIT_NUM]
                                    batch_val_label = validate_label[s*SPLIT_NUM:(s+1)*SPLIT_NUM]

                                    feed_dict = {img_placeholer: batch_val_img}
                                    [va_logits_sep] = sess.run([validate_logits], feed_dict = feed_dict)

                                    all_val_logits += va_logits_sep.tolist()
                                    all_val_labels +=  batch_val_label.tolist()


                                feed_dict = {val_input_logits_pl:all_val_logits,
                                            val_input_labels_pl:all_val_labels}
                                summary_val, val_acc_out, val_loss_out = sess.run([merged_val, validate_accuracy, validate_loss], feed_dict = feed_dict)
                                sess.run(metrics_init_op)
                                writer.add_summary(summary_val,g_step)

                                print('val_accuracy: %.3f, val_presion: %.3f, val_recall: %.3f, val_auc: %.4f,val_loss: %.3f'
                                        % (val_acc_out[0],val_acc_out[1],val_acc_out[2], val_acc_out[3], val_loss_out))


                                if val_acc_out[0] > 0.97 or loop == 0:
                                    show_flag = "{:.3f}".format(val_acc_out[0]) + \
                                            '_auc:' + "{:.4}".format(val_acc_out[3]) +\
                                            '_loss:'+  "{:.3f}".format(val_loss_out) +\
                                            '_presion:' + "{:.3f}".format(val_acc_out[1]) +\
                                            '_recall:' + "{:.3f}".format(val_acc_out[2])
                                    save_model(saver, sess, log_dir, g_step, flag = show_flag)


                    coord.request_stop()
                    coord.join(threads)




if __name__=='__main__':
    train_cls()



#                              out_imgs, out_labels = sess.run([images, labels])
                            #  for i in range(4):
                                #  c_count += 1
                                #  if out_labels[i] == 0:
                                    #  cv2.imwrite('/home/ztq/show/0_bmp/' + str(c_count)+'.bmp', out_imgs[i])

                                #  if out_labels[i] == 1:
                                    #  cv2.imwrite('/home/ztq/show/1_bmp/' + str(c_count)+'.bmp', out_imgs[i])



#  pos_wrong_dict = dict()
#  neg_wrong_dict = dict()
#  pos_wrong_dict['ALL'] = 0
#  neg_wrong_dict['ALL'] = 0
#  def cal_acc(all_gt_labels, all_predict_labels, all_val_img_roots):

    #  global pos_wrong_dict
    #  global neg_wrong_dict

    #  pos_wrong_dict['ALL'] += 1
    #  neg_wrong_dict['ALL'] += 1
    #  all_pos_count = 0
    #  all_pos_true_count = 0
    #  all_neg_count = 0
    #  all_neg_true_count = 0
    #  pos_wrong_file = open(log_dir + 'pos_wrong.txt', 'w')
    #  neg_wrong_file = open(log_dir + 'neg_wrong.txt', 'w')
    #  for ind in range(len(all_gt_labels)):

        #  if all_gt_labels[ind] == 1:
            #  all_pos_count +=1
            #  if all_predict_labels[ind] == 1:
                #  all_pos_true_count += 1
            #  else:
                #  if pos_wrong_dict.has_key(all_val_img_roots[ind]):
                    #  pos_wrong_dict[all_val_img_roots[ind]] += 1
                #  else:
                    #  pos_wrong_dict[all_val_img_roots[ind]] = 1
                #  pos_wrong_file.write(all_val_img_roots[ind] +'####' + str(pos_wrong_dict[all_val_img_roots[ind]]) + '\n')


        #  if all_gt_labels[ind] == 0:
            #  all_neg_count +=1
            #  if all_predict_labels[ind] == 0:
                #  all_neg_true_count += 1
            #  else:
                #  if neg_wrong_dict.has_key(all_val_img_roots[ind]):
                    #  neg_wrong_dict[all_val_img_roots[ind]] += 1
                #  else:
                    #  neg_wrong_dict[all_val_img_roots[ind]] = 1
                #  neg_wrong_file.write(all_val_img_roots[ind] + '####' + str(neg_wrong_dict[all_val_img_roots[ind]]) + '\n')


    #  pos_acc = float(all_pos_true_count) / all_pos_count
    #  neg_acc = float(all_neg_true_count) / all_neg_count
    #  all_val_acc = float(all_pos_true_count + all_neg_true_count) / (all_pos_count + all_neg_count)


    #  print('all_val_acc: ',str(all_val_acc)[:5],'pos_acc:',str(pos_acc)[:5], 'neg_acc:',str(neg_acc)[:5])

    #  pos_wrong_file.write(str(pos_acc)[:5] + '\n')
    #  neg_wrong_file.write(str(neg_acc)[:5] + '\n')
    #  pos_wrong_file.close()
    #  neg_wrong_file.close()

    #  sorted_pos_wrong_dict = sorted(pos_wrong_dict.items(), key=operator.itemgetter(1), reverse = True)
    #  sorted_neg_wrong_dict = sorted(neg_wrong_dict.items(), key=operator.itemgetter(1), reverse = True)
    #  pos_dict_wrong_file = open(log_dir + 'pos_wrong_dict.txt', 'w')
    #  neg_dict_wrong_file = open(log_dir + 'neg_wrong_dict.txt', 'w')
    #  for ind, item in enumerate(sorted_pos_wrong_dict):
        #  pos_dict_wrong_file.write(item[0] + '####' + str(item[1]) + '\n')
    #  for ind, item in enumerate(sorted_neg_wrong_dict):
        #  neg_dict_wrong_file.write(item[0] + '####' + str(item[1]) + '\n')
    #  pos_dict_wrong_file.close()
    #  neg_dict_wrong_file.close()

    #  return all_val_acc, pos_acc, neg_acc


#                              if True:
                                #  run_metadata = tf.RunMetadata()
                                #  _, g_step= sess.run([train_op, global_step],  \
                                    #  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),  \
                                    #  run_metadata=run_metadata)
                                #  tl = timeline.Timeline(run_metadata.step_stats)
                                #  ctf = tl.generate_chrome_trace_format()
                                #  with open('/home/public/antispoofing/other/timeline.json', 'w') as f:
                                    #  f.write(ctf)
