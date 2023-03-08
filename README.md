
# Discovering Predictable Latent Factors for Time Series Forecasting


This repository is the official implementation of "Beyond Intrinsic Modes: Discovering Predictable Latent Factors for Financial Time Series Forecasting". 



Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements and the [Qlib](https://github.com/microsoft/qlib) toolkit:

```setup
# Install the requirements
pip install -r requirements.txt

# Install Qlib
pip install --upgrade  cython
git clone https://github.com/microsoft/qlib.git && cd qlib
python setup.py install

# Download the stock features of Alpha360 from Qlib
python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn --version v2
```

## Training

### Ours+LR
```train
# CSI100
python train.py --data_set csi100 --K 0

# CSI300
 python train.py --data_set csi300 --K 0
```

### Ours+HIST
```train
# CSI100
python train.py --data_set csi100

# CSI300
 python train.py --data_set csi300
```

## Evaluation

To evaluate my model on ImageNet, run:

```eval
# Ours+LR
python eval.py --model_path <path_to_model> --K 0 --data_set <csi100/csi300>

# Ours+HIST
python eval.py --model_path <path_to_model> --data_set <csi100/csi300>
```


## Contributing

The framework of our code is based on the code in "[The HIST framework for stock trend forecasting](https://github.com/Wentao-Xu/HIST)" of the work "[HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information](https://arxiv.org/pdf/2110.13716.pdf)" by Xu et al. (WWW 2022).

