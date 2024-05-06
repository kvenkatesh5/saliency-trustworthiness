# Standard
import os

import pandas as pd
# PyTorch
import torch
import torch.nn as nn
import torchvision
# Image handling
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MURADataset(Dataset):
    def __init__(self,
                 images_file=None,  # txt file with each image name on a separate line
                 dataset=None,  # dataset name, eg: mura
                 images_dir=None,  # path to dir containing images
                 image_paths_file=None,  # txt file containing image paths from images_dir
                 labels_file=None,  # csv file containing binary labels for each image
                 labels=[],  # list of labels
                 transform=None,  # transforms to apply
                 op='train',  # train/val/test
                 img_size=299,  # image size
                 return_index=False  # retrieve indices of images
                 ):

        # Set attributes
        self.images_file = images_file
        self.images_dir = images_dir
        self.labels_file = labels_file
        self.labels = labels
        self.op = op
        self.img_size = img_size
        self.dataset = dataset
        self.return_index = return_index

        print('\n{} set: starting load'.format(self.op.capitalize()))

        # Get image paths
        all_files = []
        if image_paths_file is None:  # If no paths file, get image paths in images_dir
            for r, d, f in os.walk(images_dir):
                for filed in f:
                    if ('.png' in filed) | ('.jpg' in filed):
                        all_files.append(os.path.join(r, filed))
        else:  # If paths file, use it
            print('Using image path file')
            with open(os.path.join(self.dataset, image_paths_file), 'r') as f:
                for i in f.readlines():
                    all_files.append(i.strip('\n'))

        # Get list of images to include
        self.image_data = pd.read_table(images_file, header=None)
        self.image_data.columns = ['file']

        # Link full path for each image
        source_path = pd.DataFrame.from_dict({'path': all_files})
        source_path['file'] = source_path['path'].apply(lambda x: x[x.rfind('/') + 1:])
        self.image_data = pd.merge(self.image_data, source_path, how='left', on='file')

        # Get labels and merge into image data df
        label_data = pd.read_csv(self.labels_file)
        self.image_data = pd.merge(self.image_data, label_data, left_on='file', right_on='Image').drop(
            columns=['Image'])
        self.image_data.set_index('file', inplace=True)

        # Set transforms
        if transform == 'hflip':
            print('Using hflip transform')
            self.tfms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        elif transform == 'rotate':
            print('Using rotation transform (range: 30 degrees)')
            self.tfms = transforms.Compose([transforms.RandomRotation(30), transforms.ToTensor()])
        elif transform == 'hflip-rotate':
            print('Using hflip transform')
            print('Using rotation transform (range: 30 degrees)')
            self.tfms = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomRotation(30), transforms.ToTensor()])
        elif transform == 'hvflip-rotate':
            print('Using hvflip transform')
            print('Using rotation transform (range: 30 degrees)')
            self.tfms = transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                 transforms.RandomRotation(30), transforms.ToTensor()])
        else:
            print('Using no transforms')
            self.tfms = transforms.Compose([transforms.ToTensor()])

        # Combine transforms with standard transforms
        self.tfms = transforms.Compose([self.tfms,
                                        transforms.Resize(self.img_size),
                                        transforms.CenterCrop(self.img_size),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        print('Loaded {} images'.format(self.image_data.shape[0]))

    def __len__(self):
        return self.image_data.shape[0]

    # Gets tensor vector of labels for given image file name
    def getLabel(self, image_file):
        label = torch.zeros(len(self.labels))
        if self.image_data.loc[image_file, 'Positive'] == 1:
            label[1] = 1
        else:
            label[0] = 1
        return label

    # Get study id for given image index
    def getStudyName(self, idx):
        fileName = self.image_data.index[idx]
        return fileName[fileName.find("_") + 1:fileName.find("_image")]

    def getRegion(self, x):
        if 'SHOULDER' in x:
            return 'shoulder'
        elif 'HUMERUS' in x:
            return 'humerus'
        elif 'ELBOW' in x:
            return 'elbow'
        elif 'FOREARM' in x:
            return 'forearm'
        elif 'WRIST' in x:
            return 'wrist'
        elif 'HAND' in x:
            return 'hand'
        elif 'FINGER' in x:
            return 'finger'

    def __getitem__(self, idx):

        # Get image
        img = Image.open(self.image_data.iloc[idx]['path']).convert('RGB')

        # Image transforms
        img = self.tfms(img)

        # Get label
        label = self.getLabel(self.image_data.index[idx])

        # Test mode
        if self.op == 'test':
            region = self.getRegion(self.image_data.iloc[idx]['path'])
            f = self.image_data.iloc[idx]['path']
            study = f[:f.find('_image')]
            return img, label, study, region, f

        if self.return_index:
            # Return image, label, and index
            return img, label, torch.tensor([idx])
        else:
            # Return image and label
            return img, label


def MURAmodel(model_args):
    # Load pre-trained model
    model = None

    if model_args['model_type'] == 'InceptionV3':
        # InceptionV3
        model = torchvision.models.inception_v3(pretrained=model_args['pretrained'])
        model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=model_args['n_labels']), nn.Softmax())
        model.AuxLogits.fc = nn.Sequential(nn.Linear(in_features=768, out_features=model_args['n_labels']), nn.Softmax())

        # Freeze all but classifier and AuxLogits classifier
        if model_args['frozen']:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True
            for p in model.AuxLogits.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    elif model_args['model_type'] == 'DenseNet121':
        # DenseNet-121
        model = torchvision.models.densenet121(pretrained=model_args['pretrained'])
        model.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=model_args['n_labels']), nn.Softmax())

        # Freeze all but classifier
        if model_args['frozen']:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True
    
    elif model_args['model_type'] == 'DenseNet169':
        # DenseNet-169
        model = torchvision.models.densenet169(pretrained=model_args['pretrained'])
        model.classifier = nn.Sequential(nn.Linear(in_features=1664, out_features=model_args['n_labels']), nn.Softmax())

        # Freeze all but classifier
        if model_args['frozen']:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.classifier.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    return model
