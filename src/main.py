import os
from models import resnet
from visualization import cnn_layer_visualization, deep_dream, generate_class_specific_samples, vanilla_backprop, grad_times_image, gradcam, guided_backprop, guided_gradcam, inverted_representation, layer_activation_with_guided_backprop, integrated_gradients, smooth_grad
from utils import model_utils
import cv2

def test_layer_visualization(cnn_layer, filter_pos, dst, model):
    dst = os.path.join(dst, 'layers')
    visualizator = cnn_layer_visualization.CNNLayerVisualization(model, cnn_layer, filter_pos, dst)

    visualizator.visualise_layer_with_hooks()

def test_deep_dream(cnn_layer, filter_pos, dst, model, img):
    dst = os.path.join(dst, 'deep_dream')
    visualizator = deep_dream.DeepDream(model, cnn_layer, filter_pos, dst)

    visualizator.dream(img)

def test_generate_class_specific_samples(model, dst, target_cls):
    dst = os.path.join(dst, 'generate_clss_samples')
    visual = generate_class_specific_samples.ClassSpecificImageGeneration(model, dst)
    visual.generate(target_cls)

def test_grad_times_image(model, dst, img, target_cls):
    dst = os.path.join(dst, 'grad_times_image')
    grad_times_image.grad_times_image(model, img, target_cls, dst, 'grad')

def test_gradcam(model, img, target_cls, target_layer, dst):
    dst = os.path.join(dst, 'grad_cam')
    gradcam.grad_cam_save(model, img, target_cls, target_layer, dst, 'gradcam')

def test_guided_backprop(model, img, target_cls, dst):
    dst = os.path.join(dst, 'guide_backprop')
    guided_backprop.guided_backprop(model, img, target_cls, dst, 'guide_backprop')

def test_guide_grad_cam(model, img, target_cls, target_layer, dst):
    dst = os.path.join(dst, 'guide_grad_cam')
    guided_gradcam.guide_grad_cam_test(model, img, target_cls, target_layer, dst, 'guide_backprop')

def test_layer_activation_with_backprop(model, img, target_cls, target_layer, target_pos, dst):
    dst = os.path.join(dst, 'activation_backprop')
    layer_activation_with_guided_backprop.layer_activation_guided_backprop_test(model, img, target_layer, target_pos, target_cls, dst, "activation_backprop")

def test_inverted_prepresentation(model, img, target_cls, target_layer, dst):
    from PIL import Image
    from utils.misc import preprocess_image
    img = Image.open(img).convert("RGB")
    dst = os.path.join(dst, 'inverted_representation')

    img = preprocess_image(img)

    inverted_re = inverted_representation.InvertedRepresentation(model, dst)
    inverted_re.generate_inverted_image_specific_layer(img, img.size()[-1], target_layer)

def test_integrated_gradients(model, img, target_cls, dst):
    dst = os.path.join(dst, "integraded_gradients")
    integrated_gradients.integraded_gradients_test(model, img, target_cls, dst, "integraded_gradients")

def test_vanilla_backprop(model, dst, img, target_cls):
    dst = os.path.join(dst, 'vanilla_backprop')
    vanilla_backprop.vannilla_back(model, img, target_cls, dst, 'vanilla')

def test_smooth_grad(model, img, target_cls, dst):
    desc = 'smooth_grad'
    dst = os.path.join(dst, desc)
    smooth_grad.smooth_grad_test(model, img, target_cls, dst, 50, 4, desc)

def main():
    cnn_layer = 6
    filter_pos = 5
    pt = '/home/altas/Downloads/pts/resnet50-19c8e357.pth'
    dst = '/home/altas/altas/experiments/cnn_visualization/result/resnet50'

    filename = '/home/altas/Pictures/500.png'
    target_cls = 2
    target_layer = 6
    model = resnet.resnet50()
    model = model_utils.load_state_dict(model, pt)

    #test_layer_visualization(cnn_layer, filter_pos, dst, model)
    #test_deep_dream(cnn_layer, filter_pos, dst, model, filename)
    #test_generate_class_specific_samples(model, dst, 10)
    #test_grad_times_image(model, dst, filename, target_cls)
    #test_gradcam(model, filename, target_cls, target_layer, dst)
    #test_guided_backprop(model, filename, target_cls, dst)
    #test_guide_grad_cam(model, filename, target_cls, target_layer, dst)
    #test_inverted_prepresentation(model, filename, target_cls, target_layer, dst)
    #test_layer_activation_with_backprop(model, filename, target_cls, target_layer, filter_pos, dst)
    #test_integrated_gradients(model, filename, target_cls, dst)
    test_smooth_grad(model, filename, target_cls, dst)

    #test_vanilla_backprop(model, dst, filename, target_cls)

if __name__ == '__main__':
    main()