
import torch, patoolib, os, glob
import scipy.io
import numpy as np
import torch.utils.data as data
from typing import Optional, Callable
from torchvision.datasets.utils import download_url, _extract_zip

class PU(data.Dataset):
    #Aquí se dejan variables
    # de archivos y carpetas  
    # Archive Name 
    name_archive = ["K001.rar","K002.rar","K003.rar","K004.rar","K005.rar","K006.rar","KA01.rar","KA03.rar","KA04.rar","KA05.rar","KA06.rar","KA07.rar","KA08.rar","KA09.rar","KA15.rar","KA16.rar","KA22.rar","KA30.rar","KB23.rar","KB24.rar","KB27.rar","KI01.rar","KI03.rar","KI04.rar","KI05.rar","KI07.rar","KI08.rar","KI14.rar","KI16.rar","KI17.rar","KI18.rar","KI21.rar"]

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
        base = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"
        for name in self.name_archive:
            download_url(base + name , root=self.root) # Download
            patoolib.extract_archive( os.path.join(self.root,name)  , outdir=os.path.join(self.root) )        # Unrar
            os.remove(os.path.join(self.root,name) )  
        
    def _load_data(self):
        files_mat = glob.glob(os.path.join(self.root, "**", "*.mat"),recursive=True)    
        # 0: BaseLine, 1:Outer, 2:Inner 3:Outer+Inner
        dict_damage = {"K001": 0, "K002": 0, "K003": 0, "K004": 0, "K005": 0, "K006": 0, "KA01": 1, "KA03": 1,
               "KA05": 1, "KA06": 1, "KA07": 1, "KA08": 1, "KA09": 1, "KI01": 2, "KI03": 2, "KI05": 2,
               "KI07": 2, "KI08": 2, "KA04": 3, "KA15": 3, "KA16": 3, "KA22": 3, "KA30": 3, "KB23": 3,
               "KB24": 3, "KB27": 3, "KI04": 3, "KI14": 3, "KI16": 3, "KI17": 3, "KI18": 3, "KI21": 3}
        bearing_test = [dict_damage[val] for val in [val.split(os.sep)[1] for val in files_mat]]
                
        data_total = ()
        target_total = ()
        
        #files_mat = files_mat[:2] ## BORRAR
        
        for index, file_mat in enumerate(files_mat):
            try:
                tmp_data = scipy.io.loadmat(file_mat)[file_mat[:-4].split(os.sep)[-1]]
                tmp_data = tmp_data[0][0][2][0][6][2][0]
                tmp_data = torch.from_numpy(tmp_data).ravel()
                data = tmp_data.unfold(0, self.new_length, self.overlap).unbind()
                target = torch.Tensor([bearing_test[index]])
                if target == 3:
                    continue
                data_total = data_total + data
                target_total = target_total + target.repeat(len(data),1).unbind() 
            except: 
                print("Error Loading: ",file_mat)
                
        return data_total, target_total
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            seq = self.transform( seq )
        return seq, label
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __repr__(self) -> str:
        return "PU Dataset  (" + str(len(self.data)) + " samples) " + "0: BaseLine, 1:Outer, 2:Inner"
    
    def _get_numpy(self):
        return np.array([np.array(data[0]) for data in self]), np.array([np.array(data[1]) for data in self]).reshape(-1)