import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt


#  defining ensemble class
class EnsembleModels:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2

    def priority(self, image_path):
        with torch.no_grad():
            #  setting model states
            self.model_1.eval()
            self.model_2.eval()

            #  reading image from path
            image = cv2.imread(image_path)
            #  resizing image
            image_75 = cv2.resize(image, (75, 75))
            image_100 = cv2.resize(image, (100, 100))
            #  converting images to tensor
            image_75 = transforms.ToTensor()(image_75)
            image_100 = transforms.ToTensor()(image_100)

            #  making predictions
            output_1 = self.model_1(image_75)
            output_2 = self.model_2(image_100)

            #  deriving probabilities
            output_1p = F.softmax(F.softmax(output_1, dim=1), dim=1) * 100
            output_2p = F.softmax(F.softmax(output_2, dim=1), dim=1) * 100

            #  determining predicted class
            output_1x = torch.argmax(output_1, dim=1)
            output_2x = torch.argmax(output_2, dim=1)

            #  class dictionary
            class_dict = {0: 'sedan', 1: 'coupe', 2: 'suv', 3: 'truck'}

            #  priority logic
            if output_1x.item() in [0, 2] and output_2x.item() not in [1, 3]:
                output_3 = output_1x.item()
                return f"""
                        sedan: {output_1p[0][0].round()}% coupe: {output_1p[0][1].round()}%
                        suv: {output_1p[0][2].round()}% truck: {output_1p[0][3].round()}%   
                        Class: {class_dict[output_3].title()}
                        """
            else:
                output_3 = output_2x.item()
                return f"""
                sedan: {output_2p[0][0].round()}% coupe: {output_2p[0][1].round()}%
                suv: {output_2p[0][2].round()}% truck: {output_2p[0][3].round()}%   
                Class: {class_dict[output_3].title()}
                """

    def average_confidence(self, image_path):
        with torch.no_grad():
            #  setting model states
            self.model_1.eval()
            self.model_2.eval()

            #  reading image from path
            image = cv2.imread(image_path)
            #  resizing image
            image_75 = cv2.resize(image, (75, 75))
            image_100 = cv2.resize(image, (100, 100))
            #  converting images to tensor
            image_75 = transforms.ToTensor()(image_75)
            image_100 = transforms.ToTensor()(image_100)

            #  making predictions
            output_1 = self.model_1(image_75)
            output_2 = self.model_2(image_100)

            #  creating placeholder tensor
            placeholder = torch.ones((2, 4))
            #  inputing tensors
            placeholder[0] = output_1
            placeholder[1] = output_2

            #  computing average
            output_3 = placeholder.mean(dim=0)

            #  deriving probabilities
            output_3p = F.softmax(F.softmax(output_3, dim=0), dim=0) * 100

            #  deriving predicted class
            idx_p = torch.argmax(output_3, dim=0)

            #  class dictionary
            class_dict = {0: 'sedan', 1: 'coupe', 2: 'suv', 3: 'truck'}

            #  printing to screen
            return f"""
            sedan: {output_3p[0].round()}% coupe: {output_3p[1].round()}% suv: {output_3p[2].round()}% 
            truck: {output_3p[3].round()}%   
            Class: {class_dict[idx_p.item()].title()}
            """


#  building neural network (100px with batchnorm)
class CarRecognition100(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(8192, 514)
        self.fc2 = nn.Linear(514, 128)
        self.fc3 = nn.Linear(128, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.batchnorm_conv1 = nn.BatchNorm2d(32)
        self.batchnorm_conv2 = nn.BatchNorm2d(32)
        self.batchnorm_conv3 = nn.BatchNorm2d(64)
        self.batchnorm_conv4 = nn.BatchNorm2d(64)
        self.batchnorm_conv5 = nn.BatchNorm2d(128)
        self.batchnorm_conv6 = nn.BatchNorm2d(128)
        self.batchnorm_conv7 = nn.BatchNorm2d(128)
        self.batchnorm_fc1 = nn.BatchNorm1d(514)
        self.batchnorm_fc2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 3, 100, 100).float()
        x = F.relu(self.batchnorm_conv1(self.conv1(x)))
        x = self.pool2(F.relu(self.batchnorm_conv2(self.conv2(x))))
        x = F.relu(self.batchnorm_conv3(self.conv3(x)))
        x = self.pool4(F.relu(self.batchnorm_conv4(self.conv4(x))))
        x = F.relu(self.batchnorm_conv5(self.conv5(x)))
        x = F.relu(self.batchnorm_conv6(self.conv6(x)))
        x = self.pool7(F.relu(self.batchnorm_conv7(self.conv7(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm_fc1(self.fc1(x)))
        x = F.relu(self.batchnorm_fc2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


#  building neural network (75px with batchnorm)
class CarRecognition75(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(2048, 514)
        self.fc2 = nn.Linear(514, 128)
        self.fc3 = nn.Linear(128, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.batchnorm_conv1 = nn.BatchNorm2d(32)
        self.batchnorm_conv2 = nn.BatchNorm2d(32)
        self.batchnorm_conv3 = nn.BatchNorm2d(64)
        self.batchnorm_conv4 = nn.BatchNorm2d(64)
        self.batchnorm_conv5 = nn.BatchNorm2d(128)
        self.batchnorm_conv6 = nn.BatchNorm2d(128)
        self.batchnorm_conv7 = nn.BatchNorm2d(128)
        self.batchnorm_fc1 = nn.BatchNorm1d(514)
        self.batchnorm_fc2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 3, 75, 75).float()
        x = F.relu(self.batchnorm_conv1(self.conv1(x)))
        x = self.pool2(F.relu(self.batchnorm_conv2(self.conv2(x))))
        x = F.relu(self.batchnorm_conv3(self.conv3(x)))
        x = self.pool4(F.relu(self.batchnorm_conv4(self.conv4(x))))
        x = F.relu(self.batchnorm_conv5(self.conv5(x)))
        x = F.relu(self.batchnorm_conv6(self.conv6(x)))
        x = self.pool7(F.relu(self.batchnorm_conv7(self.conv7(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm_fc1(self.fc1(x)))
        x = F.relu(self.batchnorm_fc2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
