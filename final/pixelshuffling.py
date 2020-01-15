
import numpy as np
import torch


def pixel_shuffle_masks(n, im_dim, single):
    """ Makes pmix masks
    Args:
        n: int, the number of masks to make (one for each unique label)
        im_dim: tuple, width and height of images and thus masks
        single: bool, when activated this function will return the same mask for each label
    Return:
        np.array of shape [n, im_dim[0], im_dim[1]], the list of masks for respective labels
    """
    masks = np.zeros((n, *im_dim), dtype=int)
    for i in range(masks.shape[0]):
        if single and i > 0:
            masks[i] = masks[0]
            continue
        rrange = np.arange(0, im_dim[0]*im_dim[1])
        np.random.shuffle(rrange)
        masks[i] = rrange.reshape(im_dim)
        #masks[i, 0:16, :] = masks[i, 16:32, :]
    return masks

def pixel_shuffle_images(imgs, labels, masks):
    H, W = masks[0].shape
    
    # Makes empty result array
    pmixed = np.zeros(imgs.shape)
    for i in range(imgs.shape[0]):
        # Flatten mask for this label
        rs_mask = masks[labels[i]].reshape(W * H)
        # Flatten image
        rs_img = imgs[i].reshape((-1, W * H))
        # Mix pixels of img according to mask, accross each channel
        rs_result = rs_img[:, rs_mask]
    
        pmixed[i] = rs_result.reshape((-1, W, H))
    return pmixed

class PixelShuffler(torch.utils.data.Dataset):

    def __init__(self, base_set, pmix_seed, *, preset_masks=None, sample_size=None, transform=None):
        self.transform = transform        
        
        L = sample_size if sample_size else base_set.data.shape[0]
        imgs = np.array(base_set.data[:L])
        
        imgs = np.float32(imgs/255.)
        imgs = imgs.transpose((0, 3, 1, 2))
        
        labels = np.array(base_set.targets[:L])
        unique = np.unique(labels)
        n_unique = unique.shape[0]
        self.ingredients = [imgs[0]]
        self.classes = base_set.classes

        self.pmix_masks = pixel_shuffle_masks(n_unique, imgs.shape[2:4], True) if preset_masks is None else preset_masks
        self.pmixed = pixel_shuffle_images(imgs, labels, self.pmix_masks)
        self.labels = labels
        #show_np_array(mixed[0])
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.pmixed[idx]
        target = self.labels[idx]
        
        if self.transform is not None:
            img = self.transform(img).transpose(1, 0).transpose(1, 2)
            if not img.shape == (3, 32, 32):
                raise Exception()
        return img.float(), target
        