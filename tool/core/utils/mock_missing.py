def mock_missing(name):
    def init(self, *args, **kwargs):
        raise ImportError(
            f'The class {name} you tried to call is not importable; '
            f'this is likely due to it not being installed.')
    return type(name, (), {'__init__': init})