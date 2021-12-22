import os,inspect,sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt


def preprocess_image(img):
    means=tf.constant([[[0.485, 0.456, 0.406]]], dtype=tf.float32)
    stds=tf.constant([[[0.229, 0.224, 0.225]]], dtype=tf.float32)

    preprocessed_img=img.copy()[:, :, ::-1] # filter change.
    preprocessed_img=tf.divide(tf.subtract(preprocessed_img, means), stds)

    preprocessed_img_tensor=preprocessed_img[tf.newaxis, ...]
    return tf.Variable(preprocessed_img, trainable=False)

def save(mask, img, blurred, img_name):

    mask=tf.divide(tf.subtract(mask, tf.reduce_min(mask)), tf.reduce_max(mask))
    mask=1-mask
    mask=mask[0]
    heatmap=cv.applyColorMap(np.uint8(255*mask), cv.COLORMAP_JET)
    heatmap=np.float32(heatmap)/255
    cam=1.*heatmap+np.float32(img)/255
    cam=cam/np.max(cam)

    img=np.float32(img)/255
    perturbated=np.multiply(1-mask, img)+np.multiply(mask, blurred)
    perturbated=perturbated

    cv.imwrite(current_dir+'\\result'+'\\'+img_name+'_perturbated.png', np.uint8(255*perturbated))
    cv.imwrite(current_dir+'\\result'+'\\'+img_name+'_heatmap.png', np.uint8(255*heatmap))
    cv.imwrite(current_dir+'\\result'+'\\'+img_name+'_mask.png', np.uint8(255*mask))
    cv.imwrite(current_dir+'\\result'+'\\'+img_name+'_cam.png', np.uint8(255*cam))

def numpy_to_tensor(img, trainable=True):
    if len(img.shape)<3:
        output=tf.cast(img, dtype=tf.float32)[tf.newaxis, ...]
    else:
        output=img

    output=output[..., tf.newaxis]
    return tf.Variable(output)

def load_model():
    vgg=tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable=False
    return vgg

def explain(img_path):
    tv_beta=3
    learning_rate=0.1
    max_iterations=500
    l1_coeff=tf.Variable(1.)
    tv_coeff=tf.Variable(1.)

    model=load_model()
    original_img=cv.imread(img_path, 1)
    original_img=cv.resize(original_img, (224, 224))
    img=np.float32(original_img)/255
    blurred_img1=cv.GaussianBlur(img, (11, 11), 5)
    blurred_img2=np.float32(cv.medianBlur(original_img, 11))/255
    blurred_img_numpy=(blurred_img1+blurred_img2)/2
    mask_init=np.ones((28, 28), dtype=np.float32)

    img=preprocess_image(img)
    img=img[tf.newaxis, ...]
    blurred_img=preprocess_image(blurred_img2)
    mask=numpy_to_tensor(mask_init)

    upsample=tf.keras.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')

    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    target=model(img)
    category=tf.argmax(target, axis=1)

    for i in range(max_iterations):

        def loss(mask, img, tv_coeff, l1_coeff):
            noise=tf.random.normal((1, 224, 224, 3), mean=0, stddev=0.2)
            upsampled_mask=upsample(mask)
            upsampled_mask=tf.concat([upsampled_mask, upsampled_mask, upsampled_mask], axis=-1)

            perturbated_input=tf.multiply(img, upsampled_mask)+tf.multiply(blurred_img, 1-upsampled_mask)
            perturbated_input=tf.add(perturbated_input, noise)

            outputs=model(perturbated_input)

            m=mask[0, :, :, 0]

            row_diff=tf.divide(tf.abs(m[:-1, :]-m[1:, :]), 2.)
            col_diff=tf.divide(tf.abs(m[:, :-1]-m[:, 1:]), 2.)
            row_grad=tf.reduce_mean(tf.pow(row_diff, tv_beta))
            col_grad=tf.reduce_mean(tf.pow(col_diff, tv_beta))
            total_grad=row_grad+col_grad

            l=l1_coeff*tf.reduce_mean(tf.math.abs(1-mask))+tv_coeff*outputs[0, category.numpy()[0]]+total_grad
            # l=l1_coeff*tf.reduce_mean(tf.math.abs(1-mask))+outputs[0, category.numpy()[0]]+tv_coeff*total_grad
            l=tf.math.abs(l)
            return l

        loss_fn=lambda: loss(mask, img, tv_coeff, l1_coeff)
        var_list_fn=lambda: [mask, tv_coeff, l1_coeff]
        # var_list_fn=lambda: mask

        optimizer.minimize(loss_fn, var_list_fn, tape=tf.GradientTape())

    upsampled_mask = upsample(mask)
    save(upsampled_mask, original_img, blurred_img_numpy, img_name)


if __name__=='__main__':
    img_list=['bicycle', 'catdog', 'flute', 'tusker']
    for x in img_list[:2]:
        img_name=x
        img_path=current_dir+'\images'+'\\'+img_name+'.png'
        explain(img_path)
