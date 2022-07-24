
import torch
import torch.utils.data as data


class XJTU(data.Dataset):
    #Aquí se dejan variables
    # de archivos y carpetas  
    def __init__(
        self,
        root: str,
    )->None:
        self.root = root
        