from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]

def get_GSVCities():
    img_per_place=4
    min_img_per_place=4
    image_size=(224, 224)     #image size for GSV-Cities
    cities=TRAIN_CITIES
    mean_std=IMAGENET_MEAN_STD
    random_sample_from_each_place=True

    mean_dataset = mean_std['mean']
    std_dataset = mean_std['std']
    train_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_dataset, std=std_dataset),
    ])

    train_dataset = GSVCitiesDataset(
                cities=cities,
                img_per_place=img_per_place,
                min_img_per_place=min_img_per_place,
                random_sample_from_each_place=random_sample_from_each_place,
                transform=train_transform)
    return train_dataset