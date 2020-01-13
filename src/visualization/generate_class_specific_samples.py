"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

from torch.optim import SGD
from torchvision import models

from utils.misc import preprocess_image, recreate_image, save_image
from utils import file_utils

class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, dst):
        self.mean = [-0.485, -0.456, -0.406]
        self.std = [1/0.229, 1/0.224, 1/0.225]
        self.model = model
        self.model.eval()
        # Generate a random image
        #self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        # Create the folder to export images if not exists
        self.dst = dst
        file_utils.makesure_path(self.dst)

    def generate(self, target_class):
        initial_learning_rate = 6
        created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        for i in range(1, 150):
            # Process image and return variable
            self.processed_image = preprocess_image(created_image, False)
            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, target_class]
            print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            created_image = recreate_image(self.processed_image)
            if i % 10 == 0:
                # Save image
                im_path = os.path.join(self.dst, 'c_specific_iteration_'+str(i)+'.jpg')
                save_image(created_image, im_path)
        return self.processed_image


if __name__ == '__main__':
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    csig.generate()
