"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np
import os
from PIL import Image
from utils.misc import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            preprocess_image)
from visualization.gradcam import GradCam
from visualization.guided_backprop import GuidedBackprop


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def guide_grad_cam_test(model, img, target_cls, target_layer, dst, desc):
    img = Image.open(img).convert('RGB')
    prep_img = preprocess_image(img)

    # Grad cam
    gcv2 = GradCam(model, target_layer=target_layer)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_cls)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_cls)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)

    filename = os.path.join(dst, desc + '_ggrad_cam.jpg')
    save_gradient_images(cam_gb, filename)

    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    filename = os.path.join(dst, desc + 'ggrad_cam_gray.jpg')
    save_gradient_images(grayscale_cam_gb, filename)
    print('Guided grad cam completed')


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)

    # Grad cam
    gcv2 = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
    print('Guided grad cam completed')
