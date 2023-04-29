from towhee import pipeline
from tool.models.i_model import IModel

SUPPORTED_CHECKPOINTS = ['towhee/image-embedding-resnet50', 'towhee/image-embedding-resnet101']


class TowheeWrapperParameters:
    def __init__(self):
        self.model_checkpoint = SUPPORTED_CHECKPOINTS[0]
        self.batchsize = 32


class TowheeWrapper(IModel):
    parameters = TowheeWrapperParameters()

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters.model_checkpoint = kwargs["model_checkpoint"]

    @classmethod
    def get_model_text(cls):
        return "Select from: {0}".format(SUPPORTED_CHECKPOINTS)

    @classmethod
    def get_input_hint(cls):
        return "{{'model_checkpoint' : '{0}' }}".format(cls.parameters.model_checkpoint)

    @classmethod
    def image_transformation_pipeline(cls):
        return None

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    @classmethod
    def load_model(cls, device):
        return pipeline(cls.parameters.model_checkpoint, device=device)

    @classmethod
    def get_embedding(cls, model, img):
        return model(list(img))
