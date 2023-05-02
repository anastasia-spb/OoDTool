import numpy as np
import pandas as pd
from astropy.convolution import convolve, Gaussian2DKernel


def visualize_segmentation_probabilities(image, probabilities, color_map: Union[int, str, np.array] = cv2.COLORMAP_JET,
                                         alpha=0.8, inplace=True):
    """
    Arguments:
        image: изображение, на котором нужно визуализировать сегментацию;
        probabilities: вероятности принадлежности пикселей классу
        color_map: int - номер color_map в cv2 enum, например cv2.COLORMAP_JET == 2;
                   str - имя мэппинга;
                   np.array с shape [256, 3] или [256] - трёх- или одноканальный цвет для каждого класса сегментации;
        alpha: непрозрачность визуализации
        inplace: возвращать новое изображение или раскрашивать переданное;
    """
    assert len(probabilities.shape) == 2
    assert image.shape[0] == probabilities.shape[0]
    assert image.shape[1] == probabilities.shape[1]

    segmentation = apply_color_map((probabilities * 255).astype(np.uint8), color_map)
    probabilities = probabilities[:, :, np.newaxis] * alpha

    out = (image * (1 - probabilities) + probabilities * segmentation).astype(np.uint8)
    if inplace:
        image[:] = out[:]
    return out


def get_heatmap(gradients: np.ndarray):
    heatmap = np.mean(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.amax(heatmap)
    smoothed_heatmap = convolve(heatmap, Gaussian2DKernel(x_stddev=5))
    smoothed_heatmap /= np.linalg.norm(smoothed_heatmap)
    return smoothed_heatmap


def overlay_heatmap(gradients: np.array, crop_box: Box, img):
    crop_img = img[crop_box.slices()].copy()
    heatmap = cv2.resize(get_heatmap(gradients), (crop_img.shape[1], crop_img.shape[0]),
                         interpolation=cv2.INTER_AREA)
    probabilities = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
    return visualize_segmentation_probabilities(crop_img, probabilities, alpha=1.0, inplace=True)


def overlay_map(image, gradients):
    overlay_heatmap(gradients, image)


def test():
    grads_file = '/home/vlasova/Desktop/NIR/NIR/OoDTool/example_data/tool_working_dir/BalloonsBubbles/AlexNetWrapper_BalloonsBubbles_230502_185719.grads.pkl'
    grads_df = pd.read_pickle(grads_file)
    grads_df[data_types.RelativePathType.name()][0]
