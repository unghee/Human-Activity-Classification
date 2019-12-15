# Human-Activity-Classification
For this project, we used Python 3.6

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
```

## Running the project
Time-Series Method
```
python3 train_time_series.py
```

Frequency-Encoding Method
```
python3 train_freq.py
```

N-way Method
```
python3 train_n_way.py
