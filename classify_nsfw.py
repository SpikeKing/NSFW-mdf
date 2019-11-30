#!/usr/bin/env python


import cv2
import sys
import base64
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from image_utils import create_tensorflow_image_loader, VGG_MEAN
from image_utils import create_yahoo_image_loader
from model import OpenNsfwModel, InputType

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"


def image_decode(image_raw):
    """
    图像解码，base64输入
    """
    image = tf.decode_base64(image_raw)
    image = tf.decode_raw(image, tf.float32)  # 图像需要float32格式，根据不同的数据处理
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1, 224, 224, 3])
    return image


def dict_to_kv_string(data_dict):
    res = ""
    for key, value in data_dict.items():
        res += str(key)
        res += "="
        res += str(len(value))
        res += ":"
        res += str(value)
        res += "\n"
    return res


def process_img_opencv(img_np):
    """
    处理OpenCV图像，与源算法保持一致
    :param img_np: OpenCV图像
    :return: 当前图像
    """
    import numpy as np
    import skimage.io
    from PIL import Image
    from io import BytesIO

    im = Image.fromarray(img_np)
    imr = im.resize((256, 256), resample=Image.BILINEAR)

    fh_im = BytesIO()
    imr.save(fh_im, format='JPEG')
    fh_im.seek(0)

    image = (skimage.img_as_float(skimage.io.imread(fh_im, as_grey=False))
             .astype(np.float32))

    H, W, _ = image.shape
    h, w = (224, 224)

    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image[h_off:h_off + h, w_off:w_off + w, :]

    # RGB to BGR
    image = image[:, :, :: -1]

    image = image.astype(np.float32, copy=False)
    image = image * 255.0
    image -= np.array(VGG_MEAN, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    image = image.astype(np.float32)
    print('[Info] 图像格式: {}'.format(image.shape))
    return image


def model_predict(sess, model, img, img_name):
    """
    模型预测
    """
    predictions = sess.run(model.predictions,
                           feed_dict={model.input: img})

    print('[Info] 模型输入: {}'.format(model.input))
    print('[Info] 模型输出: {}'.format(model.predictions))

    print("[Info] Results for '{}'".format(img_name))
    print("[Info] \tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))


def save_model(sess, model):
    """
    存储TF模型
    """
    inputs = {"inputs": model.input}  # 输入String图像
    outputs = {"prob": model.predictions}  # 输出

    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
    signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}

    legacy_op = control_flow_ops.group(
        tf.local_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()),
        tf.tables_initializer())

    res_dir = "data/model-tf"
    print('[Info] 模型存储路径: {}'.format(res_dir))
    builder = saved_model_builder.SavedModelBuilder(res_dir)

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map=signature_map,
        legacy_init_op=legacy_op)

    builder.save()


def main(argv):
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    # args.input_file = "yyy-1.jpg"
    # args.input_file = "no-sexy.jpg"
    # args.input_file = "zzpic19597.jpg"
    args.input_file = "sexy.jpg"  # 输入图像

    print('[Info] 测试图像: {}'.format(args.input_file))
    args.image_loader = IMAGE_LOADER_YAHOO
    args.input_type = InputType.TENSOR.name.lower()
    args.model_weights = "data/open_nsfw-weights.npy"

    model = OpenNsfwModel()

    fn_load_image = None

    input_type = InputType[args.input_type.upper()]
    if input_type == InputType.TENSOR:
        if args.image_loader == IMAGE_LOADER_TENSORFLOW:
            fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
        else:
            fn_load_image = create_yahoo_image_loader()
    elif input_type == InputType.BASE64_JPEG:
        fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

    with tf.Session() as sess:
        model.build(weights_path=args.model_weights, input_type=input_type)
        sess.run(tf.global_variables_initializer())

        print('\n[Info] 原始版本')
        image = fn_load_image(args.input_file)  # 源图像处理格式
        model_predict(sess, model, image, args.input_file)  # 第2个版本

        print('\n[Info] 重写OpenCV版本')
        img_np = cv2.imread(args.input_file)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        image_v2 = process_img_opencv(img_np)
        model_predict(sess, model, image_v2, args.input_file)  # 第2个版本

        # 存储模型的逻辑
        # print('\n[Info] 存储模型')
        # save_model(sess, model)

    print('\n[Info] base64模型版本')
    img_np = cv2.imread(args.input_file)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = process_img_opencv(img_np)
    print('[Info] Img: {}'.format(img_np.shape))
    img_b64 = base64.urlsafe_b64encode(img_np)  # 转换base64
    img_tf = image_decode(img_b64)
    print('[Info] tf shape: {}'.format(img_tf.shape))
    img_np = tf.Session().run(img_tf)
    print('[Info] tf->np shape: {}'.format(img_np.shape))

    export_path = "data/model-tf"  # 模型文件

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_path)
        graph = tf.get_default_graph()
        print(graph.get_operations())
        res = sess.run('predictions:0',
                       feed_dict={'input:0': img_np})
        print('[Info] 最终结果: {}'.format(res))

    print('[Info] 性感值: {}'.format(res[0][1] * 100.0))


if __name__ == "__main__":
    main(sys.argv)
