import os.path


class OoDSettings:
    def __init__(self):
        self.working_dir = None
        self.use_cuda = True
        self.dataset_root_path = ''
        self.dataset_name = None
        self.metadata_folder = None

    def set_metadata_folder(self):
        if self.working_dir is None or self.dataset_name is None:
            return
        self.metadata_folder = os.path.join(self.working_dir, self.dataset_name)
        if not os.path.exists(self.metadata_folder):
            os.makedirs(self.metadata_folder)

    def set_from_config(self, config):
        self.dataset_root_path = config.get('dataset_root_path')
        self.working_dir = config.get('working_dir')

    def get_default_settings(self):
        return {
            "dataset_root_path": self.dataset_root_path,
            "working_dir": self.working_dir,
        }
