import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import numpy as np
import torch
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
import pickle
import cv2

object_categories = ['SK', 'M', 'NV']

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                if row[1]+row[2]+row[3] == '100':
                    label = 1
                if row[1]+row[2]+row[3] == '001':
                    label = 0
                if row[1]+row[2]+row[3] == '010':
                    label = 2
                item = (name, label)
                images.append(item)
            rownum += 1
    return images

# file_csv = 'D:\dataset\ISIC2017\Train_GT.csv'
# with open(file_csv, 'r') as f:
#     reader = csv.reader(f)
#     i = 0
#     for row in reader:
#         print('name is ', row[0])
#         print('label is ', type(row[1]), row[2], row[3])
#         if i == 3:
#             break
#         i += 1


class ISIC_2017(data.Dataset):
    def __init__(self, data, label, transform=None, target_transform=None):

        self.path_images = data
        self.transform = transform
        self.target_transform = target_transform

        file_csv = label

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset]set=%s number of classes=%d  number of images=%d' % (
            data, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path), target


    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

