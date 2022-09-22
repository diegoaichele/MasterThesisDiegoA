import os, torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url 
import scipy.io
from typing import Optional, Callable

class CWRU(data.Dataset):
    #Aquí se dejan variables
    # de archivos y carpetas  
    base_url = "https://engineering.case.edu/sites/default/files/"
    url= {
        "NO12K":{
            "0":{
                "1797" : {
                    "normal_0" : "97.mat" },
                "1772" : {
                    "normal_1" :"98.mat"},
                "1750" : {
                    "normal_2" :"99.mat"},
                "1730" : {
                    "normal_3" :"100.mat"},
                }
            },
        "DE12k" : {
            "0.007":{
                "1797":{
                    "IR007_0"  : "105.mat",
                    "B007_0"   : "118.mat",
                    "OR007@6_0": "130.mat",
                    "OR007@3_0": "144.mat",
                    "OR007@12_0": "156.mat"
                    },
                "1772":{
                    "IR007_1"  : "106.mat",
                    "B007_1"   : "119.mat",
                    "OR007@6_1": "131.mat",
                    "OR007@3_1": "145.mat",
                    "OR007@12_1": "158.mat"
                    },
                "1750":{
                    "IR007_2"  : "107.mat",
                    "B007_2"   : "120.mat",
                    "OR007@6_2": "132.mat",
                    "OR007@3_2": "146.mat",
                    "OR007@12_2": "159.mat"
                    },
                "1730":{ 
                    "IR007_3":"108.mat" , 
                    "B007_3" :"121.mat" , 
                    "OR007@6_3" : "133.mat" , 
                    "OR007@3_3": "147.mat" , 
                    "OR007@12_3": "160.mat" 
                    }
                },
            "0.014":{
                "1797":{
                    "IR014_0"   : "169.mat" ,
                    "B014_0"    : "185.mat" ,
                    "OR014@6_0" :"197.mat" ,
                    },
                "1772":{
                    "IR014_1": "170.mat",
                    "B014_1" : "186.mat" ,
                    "OR014@6_1" :"198.mat" ,
                    },
                "1750":{
                    "IR014_2":"171.mat" ,
                    "B014_2" :"187.mat" ,
                    "OR014@6_2":"199.mat" ,
                    },
                "1730":{
                    "IR014_3": "172.mat" ,
                    "B014_3" :"188.mat" ,
                    "OR014@6_3":"200.mat" 
                    }
                },
            "0.021":{
                "1797":{
                    "IR021_0" :"209.mat", 
                    "B021_0" :"222.mat",     
                    "OR021@6_0" :"234.mat",   
                    "OR021@3_0" :"246.mat", 
                    "OR021@12_0" :"258.mat",
                    },
                "1772":{
                    "IR021_1" :"210.mat",
                    "B021_1" :"223.mat",
                    "OR021@6_1" :"235.mat",
                    "OR021@3_1" :"247.mat",
                    "OR021@12_1" :"259.mat",
                    },
                "1750":{
                    "IR021_2"  :"211.mat" ,
                    "B021_2"  :"224.mat" ,
                    "OR021@6_2"  :"236.mat" ,
                    "OR021@3_2"  :"248.mat" ,
                    "OR021@12_2"  :"260.mat" ,
                    },
                "1730": {
                    "IR021_3" :"212.mat",
                    "B021_3" :"225.mat",
                    "OR021@6_3" :"237.mat",
                    "OR021@3_3" :"249.mat",
                    "OR021@12_3" :"261.mat",
                    }
                },
            "0.028":{
                "1797" :{
                    "IR028_0" :"3001.mat",
                    "B028_0" :"3005.mat",
                    },
                "1772" :{
                    "IR028_1" :"3002.mat",
                    "B028_1" :"3006.mat",
                    },
                "1750" :{
                    "IR028_2" :"3003.mat",
                    "B028_2" :"3007.mat",
                    },
                "1730" :{
                    "IR028_3" :"3004.mat",
                    "B028_3" :"3008.mat",            
                    }
                }
            },
        "DE48k":{
            "0.007":{
                "1797":{
                    "IR007_0" : "109.mat", 
                    "B007_0" : "122.mat", 
                    "OR007@6_0" : "135.mat", 
                    "OR007@3_0" : "148.mat", 
                    "OR007@12_0" : "161.mat",
                    },
                "1772":{
                    "IR007_1" : "110.mat", 
                    "B007_1" : "123.mat", 
                    "OR007@6_1" : "136.mat", 
                    "OR007@3_1" : "149.mat", 
                    "OR007@12_1" : "162.mat", 
                    },
                "1750":{
                    "IR007_2" : "111.mat", 
                    "B007_2" : "124.mat", 
                    "OR007@6_2" : "137.mat", 
                    "OR007@3_2" : "150.mat", 
                    "OR007@12_2" : "163.mat", 
                    },
                "1730": {
                    "IR007_3" : "112.mat", 
                    "B007_3" : "125.mat", 
                    "OR007@6_3" : "138.mat", 
                    "OR007@3_3" : "151.mat", 
                    "OR007@12_3" : "164.mat", 
                    }
                },
            "0.014":{
                "1797":{
                    "IR014_0" :"174.mat",
                    "B014_0" :"189.mat",
                    "OR014@6_0" :"201.mat",
                    },
                "1772" :{
                    "IR014_1" :"175.mat",
                    "B014_1" :"190.mat",
                    "OR014@6_1" :"202.mat",
                    },
                "1750": {
                    "IR014_2" :"176.mat",
                    "B014_2" :"191.mat",
                    "OR014@6_2" :"203.mat",
                    },
                "1730" :{
                    "IR014_3" :"177.mat",
                    "B014_3" :"192.mat",
                    "OR014@6_3" :"204.mat",
                    }
                },
            "0.021":{ 
                "1797":{
                    "IR021_0" :"213.mat",
                    "B021_0" :"226.mat",
                    "OR021@6_0" :"238.mat",
                    "OR021@3_0" :"250.mat",
                    "OR021@12_0" :"262.mat",
                    },
                "1772" :{
                    "IR021_1" :"214.mat",
                    "B021_1" :"227.mat",
                    "OR021@6_1" :"239.mat",
                    "OR021@3_1" :"251.mat",
                    "OR021@12_1" :"263.mat",
                    },
                "1750" :{ 
                    "IR021_2" :"215.mat",
                    "B021_2" :"228.mat",
                    "OR021@6_2" :"240.mat",
                    "OR021@3_2" :"252.mat",
                    "OR021@12_2" :"264.mat",
                    },
                "1730":{
                    "IR021_3" :"217.mat",
                    "B021_3" :"229.mat",
                    "OR021@6_3" :"241.mat",
                    "OR021@3_3" :"253.mat",
                    "OR021@12_3" :"265.mat",
                    }
                }
            },
        "FE12K":{
            "0.007":{
                "1797":{
                    "IR007_0":"278.mat",
                    "B007_0":"282.mat",
                    "OR007@6_0":"294.mat",
                    "OR007@3_0":"298.mat",
                    "OR007@12_0":"302.mat",
                    },
                "1772":{
                    "IR007_1":"279.mat",
                    "B007_1":"283.mat",
                    "OR007@6_1":"295.mat",
                    "OR007@3_1":"299.mat",
                    "OR007@12_1":"305.mat",
                    },
                "1750":{
                    "IR007_2":"280.mat",
                    "B007_2":"284.mat",
                    "OR007@6_2":"296.mat",
                    "OR007@3_2":"300.mat",
                    "OR007@12_2":"306.mat",
                    },
                "1730":{
                    "IR007_3":"281.mat",
                    "B007_3":"285.mat",
                    "OR007@6_3":"297.mat",
                    "OR007@3_3":"301.mat",
                    "OR007@12_3":"307.mat",
                    },
                },
            "0.014":{
                "1797":{
                    "IR014_0":"274.mat",
                    "B014_0":"286.mat",
                    "OR014@6_0":"313.mat",
                    "OR014@3_0":"310.mat",
                    },
                "1772":{
                    "IR014_1":"275.mat",
                    "B014_1":"287.mat",
                    "OR014@3_1":"309.mat",
                    },
                "1750":{
                    "IR014_2":"276.mat",
                    "B014_2":"288.mat",
                    "OR014@3_2":"311.mat",
                    },
                "1730":{
                    "IR014_3":"277.mat",
                    "B014_3":"289.mat",
                    "OR014@3_3":"312.mat",
                    }
                },
            "0.021":{
                "1797":{
                    "IR021_0":"270.mat",
                    "B021_0":"290.mat",
                    "OR021@6_0":"315.mat",
                    },
                "1772":{
                    "IR021_1":"271.mat",
                    "B021_1":"291.mat",
                    "OR021@3_1":"316.mat",
                    },
                "1750":{
                    "IR021_2":"272.mat",
                    "B021_2":"292.mat",
                    "OR021@3_2":"317.mat",
                    },
                "1730":{
                    "IR021_3":"273.mat",
                    "B021_3":"293.mat",
                    "OR021@3_3":"318.mat",
                    },
                }
            }
        }
    
    def __init__(
        self,
        root: str,
        download: bool = True,
        new_length: int = 500,
        transform: Optional[Callable] = None,
        type_data: list = ["NO12K", "DE12k", "DE48k", "FE12K"],
        type_damage: list = ["0", "0.007", "0.014", "0.021", "0.028"],
        type_velocity: list = ["1797", "1772", "1750", "1730"]
        
    )->None:
        self.root = root
        self.type_data = type_data
        self.type_damage = type_damage
        self.type_velocity = type_velocity
        self.new_length = new_length
        self.transform = transform
        self.label_dict={0:{},
                        1:{},
                        2:{},
                        3:{}}
        if download:
            self.download()

        self.data, self.targets = self._load_data()
        
        
    def __len__(self) -> int:
        return len(self.data)
    
    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")        

    def download(self) -> None:
        os.makedirs(self.raw_folder , exist_ok=True)

        for type_data in self.url.keys():
            if type_data in self.type_data:
                for type_damage in self.url[type_data].keys():
                    if type_damage in self.type_damage:
                        for type_velocity in self.url[type_data][type_damage].keys():
                            if type_velocity in self.type_velocity:
                            
                                for type_position in self.url[type_data][type_damage][type_velocity].keys():
                                    current_folder = os.path.join(self.raw_folder, type_data, type_damage, type_velocity,type_position)
                                    download_url(url=self.base_url + self.url[type_data][type_damage][type_velocity][type_position],root=current_folder)
                                    
                                  
    def _load_data(self) -> None:
        first = True
        for matfiles in  os.walk(self.root):
            if matfiles[-1] != []:
                label = matfiles[0].split(os.sep)
                name = label.pop()
                rpm =label.pop()
                if rpm not in self.type_velocity:
                    continue
                damage = label.pop()
                if damage not in self.type_damage:
                    continue
                type_data = label.pop()
                if type_data not in self.type_data:
                    continue
                try:
                    mat = scipy.io.loadmat( os.path.join(matfiles[0],matfiles[-1][0]) )
                    for data_key in mat.copy().keys():
                        if "DE_time" in data_key:
                            de_time = mat.pop(data_key)
                        # if "FE_time" in data_key:
                        #     fe_time = mat.pop(data_key)
                        # if "BA_time" in data_key:
                        #     ba_time = mat.pop(data_key)
                    signal_data = torch.tensor( [ de_time ] )
                    # TODO: Why all don't have FE and BA Time? 
                    # signal_data = torch.tensor([ de_time,
                    #                         fe_time,
                    #                         ba_time])
                except:
                    print(matfiles, "ERROR")
                    continue
                #signal_data = torch.squeeze(signal_data)
                total_length =  signal_data.shape[1]
                signal_data = signal_data[:,: total_length - total_length% self.new_length]
                # TODO: with de_tima, fe_time and ba_time, (-1, 3, self.new_length)
                signal_data = signal_data.reshape( (-1,1, self.new_length) )

                label_list = []
                for index, label in enumerate([name,rpm,damage,type_data]):
                    self.label_dict[index][label] = len(self.label_dict[index])
                    label_list.append( self.label_dict[index][label] )
                label_data = torch.Tensor( signal_data.shape[0]*label_list ).reshape(-1,signal_data.shape[0] ,len(label_list)).squeeze().long()

                if first:
                    first = False
                    signal_data_last = signal_data
                    label_data_last = label_data
                else:
                    signal_data_last = torch.cat((signal_data_last, signal_data),dim = 0)
                    label_data_last = torch.cat((label_data_last, label_data),dim = 0)    
                    break
                
        return signal_data_last, label_data_last
    
    def __getitem__(self, idx):
        seq = self.data[idx]
        label = self.targets[idx]
        if self.transform is not None:
            seq = self.transform( seq )
        return seq, label