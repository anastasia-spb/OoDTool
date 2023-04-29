from torchvision import transforms


class PerImageNormalization(object):
    def __call__(self, tensor_image):
        """
        :param img: tensor

        :return: normalized tensor
        """
        mean, std = tensor_image.mean([1, 2]), tensor_image.std([1, 2])
        if (std == 0).any():
            return tensor_image
        normalize = transforms.Normalize(mean=mean, std=std)
        return normalize(tensor_image)

    def __repr__(self):
        return self.__class__.__name__ + '()'
