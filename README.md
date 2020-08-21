# Human-Activity-Classification

## Publication
[Image Transformation and CNNs: A Strategy for Encoding Human Locomotor Intent for Autonomous Wearable Robots](https://ieeexplore.ieee.org/abstract/document/9134897)\
Ung Hee Lee, Justin Bi, Rishi Patel, David Fouhey, and Elliott Rouse
![Image of network architecture](https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/7083369/9133350/9134897/rouse2-3007455-large.gif)


## Dataset
We used [open-source bilateral and neuromechanical lower-limb dataset](https://figshare.com/articles/Benchmark_datasets_for_bilateral_lower_limb_neuromechanical_signals_from_wearable_sensors_during_unassisted_locomotion_in_able-bodied_individuals/5362627) for validating our intent recognitio system.

To setup for inference, click the "download all" button, extract into a directory named "Data" in the main repository, and unzip the files within the dataset.

## Environment Setup

Create the virtual enviroment and install required packages using [conda](https://www.anaconda.com/).

```
conda env create --name envname --file environments.yaml
```

For this project we used Python 3.6 on Ubuntu Linux 18.04.

## Running the project

There are different classifiers and configuration types provided in this repository. 

### Classifier type
There are three classifier types:
1. CNN-based spectrogram classifier (Frequency-Encoding, our method)
2. Feature-based classifiers (Heuristic)  
Among Feature-based classifiers you can choose either `LDA` or `SVM` by specifiying the classifiers within the function parameter `classifier` in `def run_classifier`.
3. Random Classifier.


### Configuration type
There are two configuration type: 1. Generic 2. Mode-specific. 

### Cross Validation type (subject dependencies)
There are two cross-validation type: 1. Subject Independent, 2. Subject Dependent (Leave-one-out cross validation). 

You can choose which classifier and configuration to use by running different python files. 
```
python3 [Classifier Type]/[Classifier Configuration]_[Subject Dependedcy].py
```

For example, Frequency-Encoding (spectrogram) type and generic configuration and subject independent case, run 

```
python3 Freq-Encoding/train_freq_loo.py
```
### Laterality and Modality
The laterlity and modality can be chosen changing the parameter `modes` and `sensors` in `def run_classifier`. User can specifiy one of three types of laterality `bilateral`,`ipsilateral`,`contralateral` and different combniations of sensor modalities `imu`,`emg`,`goin`. 


## Authors
Ung Hee Lee  
Mechanical Engineering  
University of Michigan  
unghee@umich.edu

Rishi Patel
Electrical and Computer Engineering    
University of Michigan  
patelris@umich.edu  

Justin Bi  
Electrical and Computer Engineering  
University of Michigan 
bijustin@umich.edu


