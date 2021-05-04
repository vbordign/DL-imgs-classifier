# Image Classification: The Impact of Weight Sharing and Auxiliary Losses
EPFL EE-559 Deep Learning: First Project, Spring 2021

## Requirements
- Python 3.7.4
- PyTorch 1.5.1 


## Structure

- **code/**: Path to all the source code
	- **data/**: folder where data is downloaded.
	- **figs/**: folder where figs are saved.
	- **models/**: folder where models are saved.
	- **stats/**: folder where performance indices are saved.
	- **generate_data.py**: Loads and preprocess the dataset. 
	- **models.py**: Contains different CNN architectures.
	- **parameters.py**: Contains the global parameters. 
	- **plots_tables.py**: Contains the code used for generating tables and figures for the report. 
	- **test.py**: Contains the main code used for training all models or evaluating the best trained model on the test set. 
	- **train_test.py**: Contains the code used for training and validating models. 
- **README.md**


## How to run test.py
To evaluate the best trained model in the test set:
```
$ cd code && python test.py
```

To train all models (using a GPU is strongly recommended):
```
$ cd code && python test.py --train
```

