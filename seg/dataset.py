import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

def pad_to_divisible_by_32(image):
    """A helper function to pad image size by 32.
    """
    h, w = image.shape[-2:]

    ## new shape
    h_new = (h + 31) // 32 * 32
    w_new = (w + 31) // 32 * 32
        
    return torch.nn.functional.pad(image, (0, w_new - w, 0, h_new - h))

class XMLParser:
    """A XML parser for loading annotated CT image.
    """
    def __init__(self, xml_path: str, labels: list[str], thickness=1):
        """Initialize a XML parser.

        Args:
            xml_path: path to .xml file.
            labels: labels for parsing.
            thickness: polyline thickness in drawing.
        """

        ## create annotation label map (label->idx)
        self.labels = {label: idx for idx, label in enumerate(sorted(labels))}

        ## parse the .xml file to a tree
        tree = ET.parse(xml_path)
        root = tree.getroot()

        ## select the images from root
        self.images = [child for child in root if child.tag == "image"]
        self.images = sorted(self.images, key=lambda x: int(x.attrib['id']))

        ## polyline thickness
        self.thickness = thickness

    def __len__(self)->int:
        """Size of the annotated images.
        """
        return len(self.images)

    def __getitem__(self, idx: int)->dict:
        """Annotation of the idx image.
        """
        return self._parse_image_annotation(self.images[idx])

    def _parse_image_annotation(self, root: ET.Element)->dict:
        """Parse the polygons/polylines information based on its annotation.

        Args:
            root: an image root in the tree.

        Return:
            annotation: annotation dictionary of the image.
        """
        annotation = root.attrib.copy()

        ## parse the annotation and save to the dict
        annotation['records'] = []

        ## initialize segmentation mask
        H = int(annotation["height"])
        W = int(annotation["width"])
        canvas = np.zeros((H, W, len(self.labels)), dtype=np.uint8)

        for child in root:
            label = child.attrib['label']

            if label not in self.labels:
                continue

            record = {}

            if child.tag == "polygon" or child.tag == "polyline":
                record["type"]   = child.tag
                record["label"]  = self.labels[label]
                record["points"] = child.attrib["points"]

                annotation['records'].append(record)

                points = self._parse_points(child.attrib['points'])

                ## draw polygon/polyline on individual channel
                label_idx = self.labels[label]
                if child.tag == "polygon":
                    _canvas = canvas[:,:,label_idx].copy()
                    canvas[:,:,label_idx] = self._draw_polygon(_canvas, points)
                else:
                    _canvas = canvas[:,:,label_idx].copy()
                    canvas[:,:,label_idx] = self._draw_polyline(_canvas, points)

        annotation["mask"] = canvas

        return annotation

    def _draw_polygon(self, canvas, points):
        """
        A helper function to draw the polygon on canvas.
        """

        canvas = cv2.fillPoly(canvas, [points], 1)

        return canvas

    def _draw_polyline(self, canvas, points):
        """
        A helper function to draw the polyline on canvas.
        """

        canvas = cv2.polylines(canvas, [points], False, 1, thickness=self.thickness)

        return canvas

    def _parse_points(self, points):
        """
        A helper function to parse points from .xml file.
        """

        points = points.split(";")

        pts = []

        for point in points:
            x, y = point.split(",")
            x, y = eval(x), eval(y)

            pts.append((x, y))

        pts = np.array(pts).reshape((-1, 1, 2)).astype(int)

        return pts


class CTImageDataset(Dataset):
    """CT image dataset. This implementation combines a list of .xml annotations as the dataset.

    Args:
        img_root: CT image path root.
        xml_paths: list of .xml parser path.
        labels: segmentation label list.
        thickness: polyline thickness in drawing.
        transform: image transform
    """
    def __init__(self, img_root, xml_paths, labels=[], thickness=2, transform=None):
    
        self.img_root= img_root
        self.parsers = [XMLParser(path, labels, thickness) for path in xml_paths]

        prefix_sum  = [0]

        for parser in self.parsers:
            prefix_sum.append(prefix_sum[-1] + len(parser))

        self.prefix_sum = prefix_sum

        ## image transform
        self.transform  = transform

    def __len__(self):
        return self.prefix_sum[-1]

    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        ## handle negative idx
        while idx < 0:
            idx += len(self)

        ## determine which parser to use
        parser_idx = self._idxSearch(idx)
        parser = self.parsers[parser_idx]

        ## determine idx for that parser
        annotation_idx = idx - self.prefix_sum[parser_idx]
        # print(f"parser_idx: {parser_idx}, annotation_idx: {annotation_idx}")

        ## retrieve image and segmentation mask
        annotation = parser[annotation_idx]

        image_path = os.path.join(self.img_root, annotation["name"])
        
        image    = cv2.imread(image_path, 0)
        image    = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  ## convert 1-channel gray image to 3-channel RGB image
        seg_mask = annotation['mask']

        ## transform to tensor
        image    = self._toTensor(image)
        seg_mask = self._toTensor(seg_mask)

        if self.transform:
            merged = torch.cat([image, seg_mask], dim=0)
            merged = self.transform(merged)
            
            image    = merged[:3, :, :]
            seg_mask = merged[3:, :, :]

        return image, seg_mask

    def _idxSearch(self, target):
        """Find index of target.
        """

        N = len(self.parsers)

        for i in range(N):
            if self.prefix_sum[i] <= target < self.prefix_sum[i+1]:
                return i

        return N-1

    def _toTensor(self, img):
        """A helper function to transform cv2 img to tensor.

        Args:
            img[H, W, C]: image loaded from OpenCV

        Return:
            tensor[C, H, W]: tensor of the image
        """

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]

        tensor = torch.from_numpy(img)

        ## from [H, W, C] to [C, H, W] shape
        tensor = tensor.permute([2, 0, 1])
        tensor = tensor.float()

        return tensor
    
    @property
    def labels(self):
        return self.parsers[0].labels