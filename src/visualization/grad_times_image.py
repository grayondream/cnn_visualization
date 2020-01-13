"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
def get_example_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = (('../input_images/snake.jpg', 56),
                    ('../input_images/cat_dog.png', 243),
                    ('../input_images/spider.png', 72))
    img_path = example_list[example_index][0]
    target_class = example_list[example_index][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = Image.open(img_path).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)

from utils.misc import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)

from visualization.vanilla_backprop import VanillaBackprop
from utils.misc import preprocess_image
from utils import file_utils
import os
from PIL import Image

# from guided_backprop import GuidedBackprop  # To use with guided backprop
# from integrated_gradients import IntegratedGradients  # To use with integrated grads
def grad_times_image(model, origin_image, target_cls, dst, desc):
    origin_image = Image.open(origin_image).convert('RGB')
    prep_img = preprocess_image(origin_image)

    vbp = VanillaBackprop(model)
    vanilla_grads = vbp.generate_gradients(prep_img, target_cls)
    grad_times_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    file_utils.makesure_path(dst)
    filename = os.path.join(dst, desc + '_Vanilla_grad_times_image_gray.jpg')

    save_gradient_images(grayscale_vanilla_grads, filename)

if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Vanilla backprop
    VBP = VanillaBackprop(pretrained_model)
    # Generate gradients
    vanilla_grads = VBP.generate_gradients(prep_img, target_class)

    grad_times_image = vanilla_grads[0] * prep_img.detach().numpy()[0]
    # Convert to grayscale
    grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
    # Save grayscale gradients
    save_gradient_images(grayscale_vanilla_grads,
                         file_name_to_export + '_Vanilla_grad_times_image_gray')
    print('Grad times image completed.')
