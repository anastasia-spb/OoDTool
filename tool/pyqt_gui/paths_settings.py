
class PathsSettings:
    def __init__(self):
        self.dataset_root_path = ''
        self.metadata_folder = ''

    def set_from_config(self, config):
        self.dataset_root_path = config.get('dataset_root_path')
        self.metadata_folder = config.get('metadata_folder')

    def get_default_settings(self):
        return {
            "dataset_root_path": self.dataset_root_path,
            "metadata_folder": self.metadata_folder,
        }
