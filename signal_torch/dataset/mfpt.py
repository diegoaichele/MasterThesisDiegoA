import glob
import os
import torch
import scipy.io
import numpy as np
import torch.utils.data as data
from typing import Optional, Callable
from torchvision.datasets.utils import download_url, _extract_zip
class MFPT(data.Dataset):
    #Aquí se dejan variables
    # de archivos y carpetas  
    url = 'https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'
    folder_name = "MFPT Fault Data Sets"
    def __init__(
        self,
        root: str,
        new_length: int = 312,
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
        download_url(self.url, root=self.root)
        _extract_zip( from_path= os.path.join(self.root, self.url.split('/')[-1]), to_path = self.root, compression=".xz") #'MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'), self.root)
        os.remove(os.path.join(self.root, self.url.split('/')[-1]))
        
        
    def _load_data(self):    
        files_mat = glob.glob(os.path.join(self.root, "**", "*.mat"),recursive=True)[:-4]   
        # 0: BaseLine, 1:Outer, 2:Inner
        damage_list = [0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2]
        data_total = ()
        target_total = ()
        
        for index, file_mat in enumerate(files_mat):
            try:
                tmp_data = scipy.io.loadmat( file_mat )['bearing']
                tmp_data = torch.from_numpy(tmp_data['gs'][0][0]).ravel()
                data = tmp_data.unfold(0, self.new_length, self.overlap).unbind()
                target = torch.Tensor([damage_list[index]])  
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
        return "MFPT Dataset  (" + str(len(self.data)) + " samples) " + "0: BaseLine, 1:Outer, 2:Inner"
    
    def _get_numpy(self):
        return np.array([np.array(data[0]) for data in self]), np.array([np.array(data[1]) for data in self]).reshape(-1)