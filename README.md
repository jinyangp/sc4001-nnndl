# NTU SC4001 - Neural Networks & Deep Learning Project

#  Project Overview
## Objective
Making use of the Flowers-102 dataset, our project aims to present a comprehensive analysis on how insights from past works can be used to develop a model that has decent performance and is computationally efficient, while retaining its generalisation capabilities. The model architectures we will explore include: ResNet, ResNeXt, DenseNet, and Vision Transformers (ViT).

## Dataset
Flowers-102 dataset, consists of 102 flower categories and was split into training, test and validation sets. Our train set and validation set consists of 1020 images each with 10 images per class, while our test set consists of 6149 images with a minimum of 20 images per class.

## Models Explored
Specifically, our project explored ResNet-101, ResNeXt-50-32x4d, DenseNet-121 and ViT-b-16, with pre-trained weight derived from ImageNet-1K dataset.

## Results
### CNN Models
| Models               | Epochs/Steps | Train Accuracy | Test Accuracy |
|:-------------------:|:------------:|:--------------:|:-------------:|
| ResNet-101          | 93/3000      |     0.9961     |     0.9357    |
| ResNeXt-50-32x4d    | 93/3000      | **0.999**      | **0.9489**    |
| DenseNet-121        | 93/3000      | 0.9902         |     0.925     |

### Models with Dilated Convolutions
| Models           | Dilation Rate | Epochs/Steps | Train Accuracy | Test Accuracy |
|:----------------:|:-------------:|:------------:|:--------------:|:-------------:|
| Dilated ResNet   | 2             | 93/3000      | 0.9941         | 0.9338        |
| Dilated ResNet   | 3             | 93/3000      | 1              | 0.9068        |
| Dilated ResNet   | 4             | 93/3000      | 1              | 0.904         |
| Dilated ResNet   | 5             | 93/3000      | 1              | 0.8992        |
| Dilated ResNeXt  | 2             | 78/2500      | 0.9971         | **0.9528**    |
| Dilated ResNeXt  | 3             | 62/2000      | 1              | 0.9187        |
| Dilated ResNeXt  | 4             | 62/2000      | 1              | 0.9042        |
| Dilated ResNeXt  | 5             | 62/2000      | 1              | 0.8976        |
| Dilated DenseNet | 2             | 54/1750      | 1              | 0.8982        |
| Dilated DenseNet | 3             | 54/1750      | 1              | 0.8981        | 
| Dilated DenseNet | 4             | 54/1750      | 1              | 0.8981        |
| Dilated DenseNet | 5             | 54/1750      | 1              | 0.8982        |

### ViT Models
| Models                  | Depth   | Number of Prompt Tokens | Epochs/Steps   | Train Accuracy   | Test Accuracy   |
|:-----------------------:|:-------:|:-----------------------:|:--------------:|:----------------:|:---------------:|
| Fine-tuned ViT          | -       | -                       | 93/3000        | 0.984            | 0.88            |
| Random Prompted ViT     | Shallow | 4                       | 93/3000        | 0.985            | 0.8794          |
| Random Prompted ViT     | Deep    | 4                       | 93/3000        | 0.984            | 0.8802          |
| Random Prompted ViT     | Shallow | 16                      | 93/3000        | 0.987            | 0.8795          |
| Random Prompted ViT     | Deep    | 16                      | 93/3000        | 0.987            | 0.8768          |
| Resampled Prompted ViT  | Shallow | 4                       | 15/500         | 1                | 0.9961          |
| Resampled Prompted ViT  | Deep    | 4                       | 7/250          | **1**            | **0.9963**      |

### Few-Shot Learning with Data Augmentation
| Models                         | FewShot | Augmented | Epochs / Steps | Train Accuracy | Test Accuracy |
|:------------------------------:|:---------:|:-----------:|:--------------:|:--------------:|:-------------:|
| Fine-tuned ViT                 | 5         | No          | 93/1500        | 1              | 0.7331        |
| Fine-tuned ViT                 | 5         | Yes         | 93/3000        | 0.938          | 0.7974        |
| Fine-tuned ViT                 | 10      	 | Yes         | 97/6250        | 0.939          | 0.8805        | 
| DenseNet                       | 5         | No          | 62/1000        | 1              | 0.8235        | 	  
| DenseNet                       | 5         | Yes         | 46/1500        | 0.9549         | 0.8467        |      
| DenseNet                       | 5         | Yes         | 46/1500        | 0.9549         | 0.8467        |   
|  | 5         | No          |        |               |       |   
|  | 5         | Yes         |        |        |        |
| Resampled Prompted Shallow ViT | 5         | No          | 15/250         | 1              | 0.9911        |   
| Resampled Prompted Shallow ViT | 5         | Yes         | 39/1250        | 0.9686         | 0.9932        |
   
## Conclusion

# Running the Project
## Installation
This project can be run using Python 3.12.
<br> To install Python dependencies, run ```pip install -r requirements.txt``` </br>

## Project Structure
The project is structured as follows:
```
.
├── configs
├── data
├── logs
├── README.md
├── requirements.txt
├── scripts
├── src
│   ├── cnn
│   ├── data
│   ├── metrics.py
│   ├── tool_get_dataset.py
│   ├── util.py
│   └── vit
├── test.py
└── train.py
```
- The 'configs' folder contains the configuration files to run the various models. 
- The 'data' folder contains the dataset used for training and testing.
- The 'logs' folder contains the logs generated during the training and testing process.
- The 'src' folder contains the source code for the various models, alongside the code to download the dataset and other utility functions.
- The 'scripts' folder contains the scripts to run the training and testing process, making use of the 'train.py' and 'test.py' files.

## Running the Project
After setting up the necessary configurations, run ```train.sh``` or ```test.sh``` to train or test the model respectively.

