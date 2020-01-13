import os
from models import resnet
from visualization import cnn_layer_visualization, deep_dream, generate_class_specific_samples, vanilla_backprop, grad_times_image, gradcam
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

def test_vanilla_backprop(model, dst, img, target_cls):
    dst = os.path.join(dst, 'vanilla_backprop')
    vanilla_backprop.vannilla_back(model, img, target_cls, dst, 'vanilla')


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
    test_gradcam(model, filename, target_cls, target_layer, dst)
    #test_vanilla_backprop(model, dst, filename, target_cls)

if __name__ == '__main__':
    main()