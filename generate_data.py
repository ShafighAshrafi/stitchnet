# %%
import torchvision
from torchvision.models import ResNet50_Weights
from src.dataset.main_dataset.read_data import read_worstcase_images

#create dataset
# %%
read_worstcase_images()

# %%
# weights = ResNet50_Weights.IMAGENET1K_V2
# preprocess = weights.transforms()

dataset_train = torchvision.datasets.ImageFolder('src/dataset/main_dataset/train')
dataset_val = torchvision.datasets.ImageFolder('src/dataset/main_dataset/test')

print('TRAIN', dataset_train)
print('TEST', dataset_val)
# %%
