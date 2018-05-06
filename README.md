# skeleton-AEs
An implementation of the paper "Estimating skeleton-based gait abnormality index by sparse deep auto-encoder"

## Requirements
* Python
* Numpy
* TensorFlow
* Scikit-learn

## Notice
* The code was implemented to directly work on [DIRO gait dataset](http://www-labs.iro.umontreal.ca/~labimage/GaitDataset/)
* Please download the [skeleton data](http://www.iro.umontreal.ca/~labimage/GaitDataset/skeletons.zip) and put the npz file into the folder **dataset**

## Usage
```
python3 main.py
```

## Example of output
Default training and test sets
```
(9, 9, 1200, 75)
Finish loading data
(6000, 17)
(4800, 17)
(38400, 17)

Training X...
length 1: AUC = 0.817
Training Y...
length 1: AUC = 0.763
Training Z...
length 1: AUC = 0.619

simple sum:
length 1: AUC = 0.748
length 20: AUC = 0.812
length 1200: AUC = 0.836

weighted sum:
length 1: AUC = 0.856
length 20: AUC = 0.911
length 1200: AUC = 0.945
```
