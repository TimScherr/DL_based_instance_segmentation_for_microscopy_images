import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset


class CellSegDataset(Dataset):
    """ Pytorch data set for instance cell nuclei segmentation """

    def __init__(self, root_dir, label_type, mode='train', transform=lambda x: x):
        """

        :param root_dir: Directory containing all created training/validation data sets.
            :type root_dir: pathlib Path object.
        :param mode: 'train' or 'val'.
            :type mode: str
        :param transform: transforms.
            :type transform:
        :return: Dict (image, cell_label, border_label, id).
        """

        self.img_ids = sorted((root_dir / mode).glob('img*.tif'))
        self.mode = mode
        self.root_dir = root_dir
        self.transform = transform
        self.label_type = label_type

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]

        img = np.squeeze(tiff.imread(str(img_id)))

        img = img[..., None]  # Channel dimension needed later (for pytorch)

        if self.label_type == 'distance':

            dist_label_id = img_id.parent / "dist_cell{}".format(img_id.name.split('img')[-1])
            dist_neighbor_label_id = img_id.parent / "dist_neighbor{}".format(img_id.name.split('img')[-1])

            dist_label = np.squeeze(tiff.imread(str(dist_label_id)).astype(np.float32))
            dist_neighbor_label = np.squeeze(tiff.imread(str(dist_neighbor_label_id)).astype(np.float32))

            dist_label = dist_label[..., None]
            dist_neighbor_label = dist_neighbor_label[..., None]

            sample = {'image': img,
                      'cell_label': dist_label,
                      'border_label': dist_neighbor_label,
                      'id': img_id.stem}

        elif self.label_type == 'dist':

            dist_label_id = img_id.parent / "dist_naylor{}".format(img_id.name.split('img')[-1])
            dist_label = np.squeeze(tiff.imread(str(dist_label_id)).astype(np.float32))
            dist_label = dist_label[..., None]
            sample = {'image': img,
                      'label': dist_label,
                      'id': img_id.stem}

        elif self.label_type == 'cell_dist':

            dist_label_id = img_id.parent / "dist_cell{}".format(img_id.name.split('img')[-1])
            dist_label = np.squeeze(tiff.imread(str(dist_label_id)).astype(np.float32))
            dist_label = dist_label[..., None]
            sample = {'image': img,
                      'label': dist_label,
                      'id': img_id.stem}

        elif self.label_type == 'adapted_border':

            mask_id = img_id.parent / "dist_cell{}".format(img_id.name.split('img')[-1])
            border_label_id = img_id.parent / "adapted_border{}".format(img_id.name.split('img')[-1])

            binary_label = (np.squeeze(tiff.imread(str(mask_id)).astype(np.float32)) > 0).astype(np.uint8)
            border_label = np.squeeze(tiff.imread(str(border_label_id)).astype(np.uint8))

            binary_label = binary_label[..., None]
            border_label = border_label[..., None]

            sample = {'image': img,
                      'cell_label': binary_label,
                      'border_label': border_label,
                      'id': img_id.stem}

        elif self.label_type in ['boundary', 'border', 'j4', 'jcell']:

            label_id = img_id.parent / "{}{}".format(self.label_type, img_id.name.split('img')[-1])
            label = np.squeeze(tiff.imread(str(label_id)).astype(np.uint8))
            label = label[..., None]
            sample = {'image': img, 'label': label, 'id': img_id.stem}

        sample = self.transform(sample)

        return sample
