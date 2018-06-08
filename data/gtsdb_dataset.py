import os
import xml.etree.ElementTree as ET

import glob
import numpy as np

from .util import read_image
from skimage import data, io, filters

class GTSDBDataset:

  def __init__(self, data_dir, split='trainval',
      use_difficult=False, return_difficult=False,
  ):

    self.all_img_paths = glob.glob(os.path.join(data_dir, '*.ppm'))
    self.gt_dir = os.path.join(data_dir, 'gt.txt')
    self.data_dir = data_dir
    self.use_difficult = use_difficult
    self.return_difficult = return_difficult
    self.label_names = VOC_BBOX_LABEL_NAMES

    self.gt = {}
    with open(self.gt_dir, 'r') as f:
      for line in f:
        name, x1, y1, x2, y2, class_id = line.strip('\n').split(';')
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        if name not in self.gt:
          self.gt[name] = []
        self.gt[name].append({'name': name, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class_id': class_id})
    print("self.gt length %d" % len(self.gt))

  def __len__(self):
    return len(self.gt.keys())

  def get_example(self, i):
    """Returns the i-th example.

    Returns a color image and bounding boxes. The image is in CHW format.
    The returned image is RGB.

    Args:
        i (int): The index of the example.

    Returns:
        tuple of an image and bounding boxes

    """
    print("GTSDBDataset.get_example()")
    dtype=np.float32
    img_name = list(self.gt.keys())[i]
    img = io.imread(os.path.join(self.data_dir, img_name))
    img = np.asarray(img, dtype=dtype) # (800, 1360, 3)
    img = img.transpose(2,0,1) # (3, 800, 1360)

    bbox = []
    label = []
    difficult = []

    for item in self.gt[img_name]:
      bbox.append([item['y1'],item['x1'],item['y2'],item['x2']])
      label.append(item['class_id'])
    bbox = np.array(bbox).astype(np.float32)
    label = np.array(label).astype(np.int32)

    # if self.return_difficult:
    #     return img, bbox, label, difficult
    print("GTSDBDataset.get_example() returns")
    print(img.shape)
    print(bbox.shape)
    print(label.shape)
    return img, bbox, label, difficult

  __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
  'speed limit 20',
  'speed limit 30',
  'speed limit 50',
  'speed limit 60',
  'speed limit 70',
  'speed limit 80',
  'restriction ends 80',
  'speed limit 100',
  'speed limit 120',
  'no overtaking',
  'no overtaking',
  'priority at next intersection',
  'priority road',
  'give way',
  'stop',
  'no traffic both ways',
  'no trucks',
  'no entry',
  'danger',
  'bend left',
  'bend right',
  'bend',
  'uneven road',
  'slippery road',
  'road narrows',
  'construction',
  'traffic signal',
  'pedestrian crossing',
  'school crossing',
  'cycles crossing',
  'snow',
  'animals',
  'restriction ends',
  'go right',
  'go left',
  'go straight',
  'go right or straight',
  'go left or straight',
  'keep right',
  'keep left',
  'roundabout',
  'restriction ends')
