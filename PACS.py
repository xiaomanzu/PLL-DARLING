import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels
import hub
from sklearn.preprocessing import LabelEncoder
# ds_train = hub.load('hub://activeloop/pacs-train')
# ds_test = hub.load('hub://activeloop/pacs-test')
# ds_val = hub.load('hub://activeloop/pacs-val')
class PACSdataset(Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
        self.len=data.shape[0]
    def __getitem__(self,index):    
        return self.data[index],self.label[index]
    def __len__(self):
        return self.len
        
def load_PACS(partial_rate, batch_size):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    # temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    temp_train=hub.load('hub://activeloop/pacs-train')
    

    data, labels ,domains= temp_train.images, temp_train.labels,temp_train.domains
    
    domain_num=len(domains)
    # labels=torch.tensor(labels,dtype=torch.long)
    # a_float = torch.tensor(a1, dtype=torch.float32)
    # get original data and labels

    # test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
    test_dataset=hub.load('hub://activeloop/pacs-test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*2, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    # set test dataloader
    
    partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    # generate partial labels
    le = LabelEncoder()
    data=le.fit_transform(data)
    labels=le.fit_transform(labels)
    print('data=',data,'labels=',labels)
    PACS_dataset=PACSdataset(data,labels)
    train_loader = torch.utils.data.DataLoader(dataset=PACS_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    
    labels=torch.from_numpy(labels)
    partial_matrix_dataset = PACS_Augmentention(data, partialY.float(), labels.float())
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return partial_matrix_train_loader,partialY,train_sampler,test_loader,train_loader,domains,domain_num


class PACS_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index

