import torch

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        seq = torch.unsqueeze( seq,0)
        return seq

# TODO: Change al Numpy to Torch
# [ ]   Normalize
# [ ]   ReSize
# [ ]   Retype
# [ ]   Random Random Gaussian
# [ ]   Scale
# [ ]   RandomScale
# [ ]   RandomCrop

# class Retype(object):
#     def __call__(self, seq):
#         return seq.astype(np.float32)

# class ReSize(object):
#     def __init__(self, size=1):
#         self.size = size
#     def __call__(self, seq):
#         seq = scipy.misc.imresize(seq, self.size, interp='bilinear', mode=None)
#         seq = seq / 255
#         return seq

class AddGaussian(object):
    # def __init__(self, std=0.01):
    #     self.std = std
        
    def __call__(self, seq):
        return seq + torch.tensor(seq ).normal_(std= 0.001) #self.std )
#        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

# class RandomAddGaussian(object):
#     def __init__(self, sigma=0.01):
#         self.sigma = sigma

#     def __call__(self, seq):
#         if np.random.randint(2):
#             return seq
#         else:
#             return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)

# class Scale(object):
#     def __init__(self, sigma=0.01):
#         self.sigma = sigma

#     def __call__(self, seq):
#         scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
#         return seq*scale_factor


# class RandomScale(object):
#     def __init__(self, sigma=0.01):
#         self.sigma = sigma

#     def __call__(self, seq):
#         if np.random.randint(2):
#             return seq
#         else:
#             scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1, 1))
#             return seq*scale_factor

# class RandomCrop(object):
#     def __init__(self, crop_len=20):
#         self.crop_len = crop_len

#     def __call__(self, seq):
#         if np.random.randint(2):
#             return seq
#         else:
#             max_height = seq.shape[1] - self.crop_len
#             max_length = seq.shape[2] - self.crop_len
#             random_height = np.random.randint(max_height)
#             random_length = np.random.randint(max_length)
#             seq[random_length:random_length+self.crop_len, random_height:random_height+self.crop_len] = 0
#             return seq

# class Normalize(object):
#     def __init__(self, type = "0-1"): # "0-1","1-1","mean-std"
#         self.type = type

#     def __call__(self, seq):
#         if  self.type == "0-1":
#             seq = (seq-seq.min())/(seq.max()-seq.min())
#         elif  self.type == "1-1":
#             seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
#         elif self.type == "mean-std" :
#             seq = (seq-seq.mean())/seq.std()
#         else:
#             raise NameError('This normalization is not included!')

#         return seq

# TODO: add function to quantize the sequence


class ExtractFeatures(object):
    def __init__(self, list_features=["mean","std","var","min","max","skew","max_min","kurtosis","root_mean_square"], fisher_kurtosis=False):
        self.list_features = list_features
        self.fisher_kurtosis = fisher_kurtosis

    def __call__(self, seq):
        mi_lista = []    
        if "mean" in self.list_features: 
            mi_lista.append( torch.mean(seq) )
        if "std" in self.list_features:
            mi_lista.append( torch.std(seq) )
        if "skew" in self.list_features:
            mi_lista.append( torch.mean(((seq-torch.mean(seq))/torch.std(seq))**3) )
        if "min" in self.list_features:
            mi_lista.append( torch.min(seq) )
        if "max" in self.list_features:
            mi_lista.append( torch.max(seq) )
        if "kurtosis" in self.list_features:
            mi_lista.append( torch.mean(((seq-torch.mean(seq))/torch.std(seq))**4) - self.fisher_kurtosis*3 )
        if "normsq" in self.list_features:
            mi_lista.append( torch.mean((seq-torch.mean(seq))**2) )
        if "var" in self.list_features:
            mi_lista.append( torch.var(seq) )
        if "rmse" in self.list_features:
            mi_lista.append( torch.sqrt(torch.mean((seq-torch.mean(seq))**2)) )
        if "dot" in self.list_features:
            mi_lista.append( torch.dot(seq,seq) )
        if "max_min" in self.list_features:
            mi_lista.append( torch.max(seq) - torch.min(seq) )
        if "root_mean_square" in self.list_features:
            mi_lista.append( torch.sqrt(torch.mean((seq-torch.mean(seq))**2)) )
        return torch.Tensor(mi_lista)
    