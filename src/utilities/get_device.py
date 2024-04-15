import torch


class Device:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.__initialized = True
    
    def get_device(self):
        return self.device