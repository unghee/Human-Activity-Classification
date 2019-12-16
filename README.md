# Human-Activity-Classification
For this project we used Python 3.6 on Ubuntu Linux 18.04.

The dataset was taken from: https://figshare.com/articles/Benchmark_datasets_for_bilateral_lower_limb_neuromechanical_signals_from_wearable_sensors_during_unassisted_locomotion_in_able-bodied_individuals/5362627

## Setup

Download the repoistory:

```
git clone git@github.com:Rishi-Patel/Human-Activity-Classification.git
```

Create the virtual enviroment and install required packages

```
python3 -m venv env
source env/bin/activate/

pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html # pytorch
pip3 install opencv-python # open-cv
pip3 install matplotlib # matplotlib
pip3 install numpy # numpy
pip3 install pandas # pandas
pip3 install scipy # scipy
pip3 install scikit-image # scikit-image
pip3 install temp # TemporaryFile
```

## Running the project
Time-Series Method
```
python3 Time-Series/train_time_series.py
```

Frequency-Encoding Method
```
python3 Freq-Encoding/train_freq.py
```

N-way Method
```
python3 N-way/train_n_way.py
