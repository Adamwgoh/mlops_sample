'''
Sample Dataset loading format for unet
'''
import os
import glob
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class DirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, scale=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.data_pair = []
        self.scale = scale

        assert Path(self.img_dir).exists(),  f"image_dir {self.img_dir} is not a valid directory"
        assert Path(self.mask_dir).exists(), f"mask_dir {self.mask_dir} is not a valid directory"
        assert scale >= 0.0, f"scale cannot be smaller than 0. Given: {self.scale}"

        self.img_paths  = [x.absolute() for x in Path(self.img_dir).iterdir() if x.is_file()]
        self.mask_paths = [x.absolute() for x in Path(self.mask_dir).iterdir() if x.is_file()]

        # make a pair and remove those that dont have a match
        for s in self.img_paths:
            assert Path(self.mask_dir, s.stem + "_mask" + s.suffix).absolute() in self.mask_paths
            mask_path = Path(self.mask_dir, s.stem + "_mask" + s.suffix).absolute()
            pair = (s, mask_path)
            self.data_pair.append(pair)

        self.ids = [x.stem for x in self.img_paths]

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img):
        w, h = img.size
        _h = int(h * self.scale)
        _w = int(w * self.scale)
        assert _w > 0
        assert _h > 0

        _img = img.resize((_w, _h))
        _img = np.array(_img)
        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        if _img.max() > 1:
            _img = _img / 255.
        return _img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_files = glob.glob(os.path.join(self.img_dir, idx+'.*'))
        mask_files = glob.glob(os.path.join(self.mask_dir, idx+'_mask.*'))

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        # use Pillow's Image to read .gif mask
        # https://answers.opencv.org/question/185929/how-to-read-gif-in-python/
        #img = Image.open(img_files[0])
        #mask = Image.open(mask_files[0])
        #assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        return torch.from_numpy(img).float(), \
            torch.from_numpy(mask).float()