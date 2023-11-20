import itertools
import math
import pathlib

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from tqdm import tqdm


def repeater(iterable):
    for it in itertools.repeat(iterable):
        for item in it:
            yield item


class L3FDataset(Dataset):

    def __init__(self, cfg, mode, memorize=False):
        super().__init__()

        assert cfg.split in ['20', '50', '100']
        assert mode in ['train', 'test']

        root = pathlib.Path(cfg.root)
        assert root.exists()

        self.gt_fpaths = sorted(root.joinpath('jpeg', mode, '1').glob('*.jpg'))
        self.lq_fpaths = sorted(root.joinpath('jpeg', mode, f'1_{cfg.split}').glob('*.jpg'))

        self.mode = mode
        self.patch_size = cfg.patch_size or None
        self.size_divisibility = cfg.size_divisibility or 1

        # crop central views images
        self.resolution = cfg.cropped_resolution or 15

        # ignore more views from the top and left boundary to align the official implementation
        self.sampled_area_start = math.ceil((15 - self.resolution) * 0.5)
        self.sampled_area_end = self.sampled_area_start + self.resolution

        self.memorize = memorize
        # memory cache, indexed by file paths
        self.lq_views_dict = {}
        self.gt_views_dict = {}

        if memorize:
            for i in tqdm(range(len(self)), desc='loading dataset into memory', leave=False):
                _ = self[i]
            print('dataset has been loaded')

    def __len__(self):
        return len(self.gt_fpaths)

    def __getitem__(self, index):
        lq_fpath, gt_fpath = self.lq_fpaths[index], self.gt_fpaths[index]
        stem = lq_fpath.stem
        if self.memorize:
            if lq_fpath not in self.lq_views_dict:
                lq_views, gt_views = self._pack_views(lq_fpath, gt_fpath)
                self.lq_views_dict[lq_fpath] = lq_views
                self.gt_views_dict[gt_fpath] = gt_views

            lq_views = self.lq_views_dict[lq_fpath]
            gt_views = self.gt_views_dict[gt_fpath]
        else:
            lq_views, gt_views = self._pack_views(lq_fpath, gt_fpath)

        if self.mode == 'train':
            # Random Cropping
            i, j, th, tw = RandomCrop.get_params(lq_views, (self.patch_size, self.patch_size))
            out_lq_views = T.crop(lq_views, i, j, th, tw)
            out_gt_views = T.crop(gt_views, i, j, th, tw)

            # Random Flipping
            if torch.rand(1) < 0.5:
                out_lq_views = T.hflip(out_lq_views)
                out_gt_views = T.hflip(out_gt_views)

            if torch.rand(1) < 0.5:
                out_lq_views = T.vflip(out_lq_views)
                out_gt_views = T.vflip(out_gt_views)
        else:
            out_lq_views = self._pad_test_image(lq_views, self.size_divisibility)
            out_gt_views = gt_views

        return {'lq': out_lq_views, 'gt': out_gt_views, 'stem': stem}

    def _pack_views(self, lq_fpath, gt_fpath):
        lq_img = T.to_tensor(Image.open(lq_fpath))
        gt_img = T.to_tensor(Image.open(gt_fpath))
        lq_views = rearrange(lq_img, 'c (n1 h) (n2 w) -> n1 n2 c h w', n1=15, n2=15)
        gt_views = rearrange(gt_img, 'c (n1 h) (n2 w) -> n1 n2 c h w', n1=15, n2=15)

        # ignore peripheral views
        lq_views = lq_views[self.sampled_area_start:self.sampled_area_end,
                            self.sampled_area_start:self.sampled_area_end]
        gt_views = gt_views[self.sampled_area_start:self.sampled_area_end,
                            self.sampled_area_start:self.sampled_area_end]

        return lq_views, gt_views

    def _pad_test_image(self, img, size_divisibility):
        h, w = img.shape[-2:]
        new_h = ((h + size_divisibility) // size_divisibility) * size_divisibility
        new_w = ((w + size_divisibility) // size_divisibility) * size_divisibility
        pad_h = new_h - h if h % size_divisibility != 0 else 0
        pad_w = new_w - w if w % size_divisibility != 0 else 0
        out = F.pad(img, [0, pad_w, 0, pad_h, *(0, 0) * (img.ndim-4)], mode='reflect')
        return out
