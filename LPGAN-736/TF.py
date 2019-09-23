import os, sys
import tensorflow as tf
import time
import numpy as np

from DATA import *
from MODEL import *
from FUNCTION import *
from PREPROCESSING import *

print(current_time() + ', exp = %s, load_model path = %s' % (FLAGS['num_exp'], os.path.dirname(os.path.abspath(__file__))))
#os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS['num_gpu']
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

netG_act_o = dict(size=1, index=0)
netG = NetInfo('netG-%d' % FLAGS['num_exp'])
test_df = DataFlow()


def checkValidImg(input_img):
    print(current_time() + ', [checkValidImg]')
    if input_img is None:
        print(current_time() + ', img is None')
        return None
    if len(input_img.shape) != 3:
        print(current_time() + ', len(shape) != 3')
        return None
    if input_img.shape[2] != 3:
        print(current_time() + ', shape[2] != 3')
        return None
    if input_img.dtype != np.uint8:
        print(current_time() + ', img.dtype != uint8')
        return None
    return True

def normalizeImage(img):
    print(current_time() + ', [normalizeImage]')
    [height, width, channels] = img.shape
    print(current_time() + ', original shape = [%d, %d, %d]' % (height, width, channels))
    max_l = max(height, width)

    is_need_resize = max_l != FLAGS['data_image_size']
    if is_need_resize:
        use_gpu = False 
        if use_gpu and is_downsample:
            # gpu
            new_h, new_w = get_normalize_size_shape_method(img)
            dict_d = [img, new_h, new_w]
            dict_t = [tf_input_img_ori, tf_img_new_h, tf_img_new_w]
            img = sess.run(tf_resize_img, feed_dict={t:d for t, d in zip(dict_t, dict_d)})
        else:
            # cpu
            img = cpu_normalize_image(img)

    return img

def getInputPhoto(file_name):
    print(current_time() + ', [getInputPhoto]: file_name = %s' % (FLAGS['folder_input'] + file_name))
    file_name_without_ext = os.path.splitext(file_name)[0]
    input_img = cv2.imread(FLAGS['folder_input'] + file_name, 1)

    cv2.imshow('enhance_test_img',input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    os.remove(FLAGS['folder_input'] + file_name)
    if checkValidImg(input_img):
        resize_input_img = normalizeImage(input_img)
        file_name = file_name_without_ext + FLAGS['data_output_ext'] 
        cv2.imwrite(FLAGS['folder_processed_input'] + file_name, resize_input_img)
        return file_name
    else:
        return None

def processImg(file_in_name, file_out_name_without_ext):
    print(current_time() + ', [processImg]: file_name = %s' % (FLAGS['folder_processed_input'] + file_in_name))
    input_img = cv2.imread(FLAGS['folder_processed_input'] + file_in_name, -1)
    print "Input shape: ",input_img.shape
    print input_img
    input_img, _, rect = random_pad_to_size(input_img, FLAGS['data_image_size'], None, True, False)
    print "Input shape: ",input_img.shape
    print input_img
    input_img = input_img[None, :, :, :]
    dict_d = [input_img, rect, 0]
    dict_t = [test_df.input1_src] + \
        test_df.mat1.rect + test_df.mat1.rot
    enhance_test_img = sess.run(netG_test_output1_crop, feed_dict={t:d for t, d in zip(dict_t, dict_d)})
    print("Enhanced image: ",enhance_test_img)
    print enhance_test_img * tf.as_dtype(FLAGS['data_input_dtype']).max
    enhance_test_img = safe_casting(enhance_test_img * tf.as_dtype(FLAGS['data_input_dtype']).max, FLAGS['data_input_dtype'])
    print "Enhanced image: ", enhance_test_img
    

    enhanced_img_file_name = file_out_name_without_ext + FLAGS['data_output_ext']
    enhance_img_file_path = FLAGS['folder_results'] + enhanced_img_file_name
    #try:
    #    print(current_time() + ', try remove file path = %s' % enhance_img_file_path)
    #    os.remove(enhance_img_file_path)
    #except OSError as e:
    #    print(current_time() + ', remove fail, error = %s' % e.strerror)
    cv2.imwrite(enhance_img_file_path, enhance_test_img)
    return enhanced_img_file_name

with tf.name_scope(netG.name):
    with tf.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_test_output1 = model(netG, test_df.input1, False, netG_act_o, is_first=True)
assert len(netG.weights) == len(netG.parameter_names), len(weights) != len(parameters)
saver = tf.train.Saver(var_list=netG.weights, max_to_keep=None)

with tf.name_scope("Loss"):
    netG_test_output1_crop = tf_crop_rect(netG_test_output1, test_df.mat1, 0)

with tf.name_scope("Resize"):
    tf_input_img_ori = tf.placeholder(tf.uint8, shape=[None, None, 3])
    tf_img_new_h = tf.placeholder(tf.int32)
    tf_img_new_w = tf.placeholder(tf.int32)
    tf_resize_img = tf.image.resize_images(images=tf_input_img_ori, size=[tf_img_new_h, tf_img_new_w], method=tf.image.ResizeMethod.AREA)

sess_config = tf.ConfigProto(log_device_placement=False)
sess_config.gpu_options.allow_growth = True

sess = tf.Session(config=sess_config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

    

with tf.Session() as sess:

    print "Min: ", np.iinfo(np.uint8).min
    print "Max: ", np.iinfo(np.uint8).max
    
    saver.restore(sess, FLAGS['load_model_path_new']) 
    ### Node list ### 
    # for n in sess.graph_def.node:
    #     print('\n',n.name)
    # exit()

    # Tensorboard
    # writer = tf.compat.v1.summary.FileWriter('.')
    # writer.add_graph(sess.graph)
    # print("Acabe")
    # exit()

    print("\n ######################################################## \n")
    print("Modelo restaurado con exitooooooo: ",FLAGS['load_model_path_new'])
    print("\n ######################################################## \n")


    if len(sys.argv) == 1:
        file_list_input = os.listdir(FLAGS['folder_input'])
        if len(file_list_input) != 0:
            for f in file_list_input:
                input_file_name = getInputPhoto(f)

        file_list_processed_input = os.listdir(FLAGS['folder_processed_input'])
        for f in file_list_processed_input:
                file_name_without_ext = os.path.splitext(f)[0]
                
                start = time.time()
                output_file_name = processImg(file_name_without_ext + '.png',file_name_without_ext + '_enhanced')
                end = time.time()
                print "Time: ",end - start
    
    if len(sys.argv) == 2:
        input_image = sys.argv[1]
        file_name_without_ext = os.path.splitext(input_image)[0]
        if os.path.isfile(FLAGS['folder_input'] + input_image) is False and os.path.isfile(FLAGS['folder_processed_input'] + file_name_without_ext + '.png') is False:
            print('Wrong image path.\nImage: '+ input_image)
            
        if os.path.isfile(FLAGS['folder_processed_input'] + file_name_without_ext + '.png'):
            start = time.time()
            output_file_name = processImg(file_name_without_ext + '.png',file_name_without_ext + '_enhanced')
            end = time.time()
            print "Time: ",end - start

        if os.path.isfile(FLAGS['folder_input'] + input_image):

            input_file_name = getInputPhoto(input_image)
            start = time.time()
            output_file_name = processImg(input_file_name,file_name_without_ext + '_enhanced')
            end = time.time()
            print "Time: ",end - start
            

   

    #input_file_name = getInputPhoto('jaleo.JPG')
    #print("\n ######################################################## \n")
    #print("Input file name: ",input_file_name)
    #print("\n ######################################################## \n")
    #output_file_name = processImg(input_file_name,'jaleo_enhanced')
    #print("\n ######################################################## \n")
    #print("Output file name: ",output_file_name)
    #print("\n ######################################################## \n")
