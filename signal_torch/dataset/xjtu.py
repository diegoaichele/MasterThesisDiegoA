
import torch, os, patoolib
import torch.utils.data as data
from typing import Optional, Callable
from torchvision.datasets.utils import  download_file_from_google_drive

# The run-to-failure data of 15 rolling element bearings are included in the data packet 
# (XJTU-SY_Bearing_Datasets.zip)
class XJTU(data.Dataset):
    #Aquí se dejan variables
    # de archivos y carpetas
    url_drive = {"XJTU-SY_Bearing_Datasets.part01.rar":"1ATvZuD6j3bPxhyR07Zm-PURmOC4b4uRn",
             "XJTU-SY_Bearing_Datasets.part02.rar":"162KvWNIpBGtd7EDWo4yP1j5XsaoNHOYU",
             "XJTU-SY_Bearing_Datasets.part03.rar":"1NvzrGW-KOSy48OZmiFxlE3TPV4CKAcw0",
             "XJTU-SY_Bearing_Datasets.part04.rar":"1VuQ5-mK11p1S2pTxUZaH_IxOwUlsmN0S",
             "XJTU-SY_Bearing_Datasets.part05.rar":"1WH4OU4MLaMGQkbh6DghxPA5Dwvsq8tEf",
             "XJTU-SY_Bearing_Datasets.part06.rar":"1wzQzQUx6-J8DuGczT81OkrkTgOUwL-I_"
                 }
    def __init__(
        self,
        root: str,
        new_length: int = 1024,
        overlap: float = 0.0,
        transform: Optional[Callable] = None,
        download: bool = True,
    )->None:
        self.root = root
        self.new_length = new_length
        self.transform = transform
        assert overlap < 1  and overlap >=0, "Values of overlap is [0,1)"
        self.overlap = int((1-overlap)*new_length)
        assert self.overlap != 0, "Value of overlap is too big"
        if download:
            self.download()
        self.data, self.targets = self._load_data()
        
    def download(self):
        for filename in self.url_drive.keys():
            download_file_from_google_drive(self.url_drive[filename],"XJTU", filename)
        patoolib.extract_archive(os.path.join(self.root,"XJTU-SY_Bearing_Datasets.part01.rar")  , outdir= "XJTU")
        for filename in self.url_drive.keys():
            os.remove(os.path.join(self.root,filename) )  