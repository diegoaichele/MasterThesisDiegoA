
import os
import torch
import scipy.io
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
        transform: Optional[Callable] = None,
        download: bool = True,
    )->None:
        self.root = root
        self.new_length = new_length
        self.transform = transform
        if download:
            self.download()
        
        self.data, self.targets = self._load_data()
        
        

    def download(self):
        download_url(self.url, root=self.root)
        _extract_zip( from_path= os.path.join(self.root, self.url.split('/')[-1]), to_path = self.root, compression=".xz") #'MFPT-Fault-Data-Sets-20200227T131140Z-001.zip'), self.root)
        os.remove(os.path.join(self.root, self.url.split('/')[-1]))
        
        
    def _load_data(self):    
        files_mat =[os.path.join(self.root,"MFPT Fault Data Sets","1 - Three Baseline Conditions","baseline_1.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","1 - Three Baseline Conditions","baseline_2.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","1 - Three Baseline Conditions","baseline_3.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","2 - Three Outer Race Fault Conditions","OuterRaceFault_1.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","2 - Three Outer Race Fault Conditions","OuterRaceFault_2.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","2 - Three Outer Race Fault Conditions","OuterRaceFault_3.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_1.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_2.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_3.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_4.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_5.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_6.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","3 - Seven More Outer Race Fault Conditions","OuterRaceFault_vload_7.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_1.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_2.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_3.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_4.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_5.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_6.mat"),
                    os.path.join(self.root,"MFPT Fault Data Sets","4 - Seven Inner Race Fault Conditions","InnerRaceFault_vload_7.mat")]
        # 0: BaseLine, 1:Outer, 2:Inner
        damage_dict = {0:0,1:0,2:0,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:2,14:2,15:2,16:2,17:2,18:2,19:2}
        
        data_total = ()
        target_total = ()
        
        for index, file_mat in enumerate(files_mat):
            tmp_data = scipy.io.loadmat( file_mat )['bearing']
            try:
                data = torch.from_numpy(tmp_data['gs'][0][0][:-(len(tmp_data["gs"][0][0])%self.new_length)]).reshape(-1,self.new_length).unbind()
            except:
                data = torch.from_numpy(tmp_data["gs"][0][0][:-(len(tmp_data["gs"][0][0])%self.new_length)]).reshape(-1,self.new_length).unbind()
            target = torch.Tensor( (damage_dict[index], int(tmp_data['sr'][0][0][0][0]), int(tmp_data['rate'][0][0][0][0]), int(tmp_data['load'][0][0][0][0]) ) )
            data_total = data_total + data
            target_total = target_total + target.repeat(len(data),1).unbind() 
        
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
        return "MFPT Dataset  (" + str(len(self.data)) + " samples)" + "0: BaseLine, 1:Outer, 2:Inner"
