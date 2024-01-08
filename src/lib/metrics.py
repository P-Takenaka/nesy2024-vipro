import torch
import torchmetrics

class LPIPS(torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity):
    higher_is_better = False

    def __init__(
            self, **kwargs):
        super().__init__(net_type='vgg', normalize=False) #normalize=False -> [-1,1] range

    def update(self, **kwargs):
        preds = torch.clip(kwargs['post_rgb_recon_combined'], -1.0, 1.0)
        target = kwargs['video']

        # Merge time and batch dim
        return super().update(img1=target.view((-1,) + target.shape[2:]), img2=preds.view((-1,) + preds.shape[2:]))


class SSIM(torchmetrics.StructuralSimilarityIndexMeasure):
    higher_is_better = True

    def __init__(
            self, **kwargs):
        super().__init__()

    def update(self, **kwargs):
        preds = kwargs['post_rgb_recon_combined']
        target = kwargs['video']

        # Merge time and batch dim
        return super().update(target=target.view((-1,) + target.shape[2:]), preds=preds.view((-1,) + preds.shape[2:]))


class PSNR(torchmetrics.PeakSignalNoiseRatio):
    higher_is_better = True

    def __init__(
            self, **kwargs):
        super().__init__()

    def update(self, **kwargs):
        preds = kwargs['post_rgb_recon_combined']
        target = kwargs['video']

        # Merge time and batch dim
        return super().update(target=target.view((-1,) + target.shape[2:]), preds=preds.view((-1,) + preds.shape[2:]))
