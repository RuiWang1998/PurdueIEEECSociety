import torchvision.transforms as transforms
import torchvision.transforms.functional as F

generic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    #transforms.CenterCrop(size=128),
    transforms.Lambda(lambda x: (48, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1, 1, 1)),
    transforms.ToPILImage(),
])

new_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0., 0., 0.), (1, 1, 1)),
    transforms.ToTensor(),
    transforms.ToPILImage()
    ])
