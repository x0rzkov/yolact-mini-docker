import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
from data.config import cfg
from pycocotools.coco import COCO
import pdb


def detection_collate(batch):
    imgs = []
    targets = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        masks.append(torch.FloatTensor(sample[2]))
        num_crowds.append(sample[3])

    return torch.stack(imgs, 0), targets, masks, num_crowds


class COCODetection(data.Dataset):
    def __init__(self, image_path, info_file, augmentation=None):
        self.image_path = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys())  # 117266 for train2017
        self.augmentation = augmentation
        self.label_map = cfg.label_map

    def __getitem__(self, index):
        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, gt, masks, num_crowds

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        ann_id = self.coco.getAnnIds(imgIds=img_id)

        # multi instances, for one instance: {'segmentation', 'area', iscrowd', 'image_id', 'bbox', 'category_id'}
        all_instances = self.coco.loadAnns(ann_id)

        # Separate out crowd annotations. These are annotations that signify a large crowd of objects, where there is
        # no annotation for each individual object. When testing and training, treat these crowds as neutral.
        crowd_instances = [aa for aa in all_instances if aa['iscrowd']]
        # Crowd instances are now at the end of the array.
        all_instances = [aa for aa in all_instances if not aa['iscrowd']] + crowd_instances

        img_name = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = osp.join(self.image_path, img_name)
        assert osp.exists(img_path), f'Can not find \'{img_path}\'.'

        original_img = cv2.imread(img_path)
        height, width, _ = original_img.shape

        if len(all_instances) > 0:
            masks = [self.coco.annToMask(aa).reshape(-1) for aa in all_instances]
            masks = np.vstack(masks).reshape((-1, height, width))  # between 0~1, (num_instances, height, width)
            # Uncomment this to visualize the masks.
            # cv2.imshow('aa', masks[0]*255)
            # cv2.waitKey()

            scale = np.array([width, height, width, height])
            all_boxes = []
            for instance in all_instances:
                assert instance['bbox'], f'bbox error for {instance}'
                bbox = instance['bbox']
                label_idx = self.label_map[instance['category_id']] - 1
                x1y1x2y2c = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                x1y1x2y2c.append(label_idx)
                all_boxes.append(x1y1x2y2c)  # (xmin, ymin, xmax, ymax, label_idx), between 0~1

        if self.augmentation:
            assert len(all_boxes) > 0, 'Not enough bbox.'
            box_array = np.array(all_boxes)

            img, masks, boxes, labels = self.augmentation(original_img, masks, box_array[:, :4],
                                                          {'num_crowds': len(crowd_instances), 'labels': box_array[:, 4]})

            # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations.
            num_crowds = labels['num_crowds']
            labels = labels['labels']
            boxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(img).permute(2, 0, 1), boxes, masks, height, width, num_crowds
