import numpy as np
import matplotlib as mpl
from astropy.convolution import convolve, Gaussian2DKernel
from PIL import Image


def visualize_probabilities(image, probabilities, alpha=0.8):
    cmap = mpl.colormaps['jet']
    segmentation = Image.fromarray(np.uint8(cmap(probabilities) * 255)).convert('RGB')
    probabilities = probabilities[:, :, np.newaxis] * alpha

    out = (image * (1 - probabilities) + probabilities * segmentation).astype(np.uint8)
    out = Image.fromarray(out)
    return out


def get_heatmap(gradients: np.ndarray):
    heatmap = np.mean(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.amax(heatmap)
    smoothed_heatmap = convolve(heatmap, Gaussian2DKernel(x_stddev=5))
    smoothed_heatmap /= np.linalg.norm(smoothed_heatmap)
    return smoothed_heatmap


def overlay_heatmap(gradients: np.array, img: Image):
    heatmap = get_heatmap(gradients)
    heatmap = Image.fromarray(np.uint8(heatmap * 255))
    heatmap = np.array(heatmap.resize(img.size))
    probabilities = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()))
    return visualize_probabilities(img, probabilities, alpha=0.7)


def overlay_map(image, grads_file):
    with open(grads_file, 'rb') as f:
        grads = np.load(f)
    im = Image.open(image).convert('RGB')
    return overlay_heatmap(grads, im)


def test():
    grads_file = '/home/vlasova/Desktop/NIR/NIR/OoDTool/example_data/tool_working_dir/BalloonsBubbles/grads/balloon/test/7308740338_591f27b631_k.jpg.grads.npy'
    image = '/home/vlasova/Desktop/NIR/NIR/OoDTool/example_data/datasets/BalloonsBubbles/balloon/test/7308740338_591f27b631_k.jpg'
    result = overlay_map(image, grads_file)
    result.show()


if __name__ == "__main__":
    test()
