# Code for classifying collision types

## Models that can be used: 
- `knn`: unsupervised learning method, clustering the data based on their nearest neighboors laying on a feature space. 
- `decision_tree`: directed graph model that aims to take a decision based on a succession of decision nodes. 
- `random_forest`: ensemble learning methods based on the agregation of multiple decision trees.
- `linear_svm`: discriminative classifier that separates the data originally into two classes, using support vectors, in the euclidean space. For multi-class classification, SVM classifies with a sucessive one-versus-all classification.
- `svm`: svm used with a kernel trick to move the data from one space to another in order to make it separable. 
- `logistic_regression`: regression obtained on log-odds successively in a one-versus-all fashion. 
- `naive_bayes`: Bayes rules applied in a naive way to the dataset.
- `adaboost`: ensemble learning method that aims to boost weak classifiers based on their failures. 

## Dataset to test: 
- `pouplin`: dataset of collisions obtained from debris disk accretion simulation of a martian debris disks over 1Myr for 5,000 particles and their expected outcomes using Leinhardt and Stewart (2012). 4996 data points.
- `cambioni`: open source dataset used for rock collisions, with four classes: (hit-and-run, graze-and-merge, merging, disruption), four features (Mass, Gamma, Angle, Velocity), from 769 points.
- `iris`: toy dataset to classify types of flowers, with three classes: (Setosa, Versicolour, Virginica), four spaces (Sepal Length, Sepal Width, Petal Length, Petal Width), from 150 points.

## Optimisation
The optimisation of the different parameters is obtained using a grid search. 

## Run the code: 
- `python main.py --model decision_tree --dataloader cambioni --optim` (example for running the model decision_tree, and optimise it on the cambioni datatset)
- `python main.py --model knn --dataloader iris` (example for running the model knn without optimisation on the iris datatset)

## Description of dataset

<img
src=“./images/pouplin.png”
raw=true
alt=“Martian Moons Dataset”
style=“margin-right: 10px;”
/>

## Results 

Using a decision tree model: 
train time: 3.661s
done in 0.000182s
accuracy:   0.917

               hit-and-run  merger  disruption  supercatastrophic
hit-and-run                    0.93    0.06        0.01               0.00
merger                           0.04    0.96        0.00               0.00
disruption                      0.22    0.05        0.51               0.22
supercatastrophic         0.00    0.01        0.11               0.88


Using knn:

train time: 92.584s
done in 0.025493s
accuracy:   0.616

              hit-and-run  merger  disruption  supercatastrophic
hit-and-run                 0.51    0.41         0.0               0.08
merger                        0.18    0.81         0.0               0.00
disruption                   0.73    0.15         0.0               0.12
supercatastrophic      0.67    0.05         0.0               0.28


Using random_forest:

train time: 35.061s
done in 0.010664s
accuracy:   0.932

              hit-and-run  merger  disruption  supercatastrophic
hit-and-run               0.93    0.07        0.00               0.00
merger                      0.03    0.97        0.00               0.00
disruption                  0.17    0.10        0.41               0.32
supercatastrophic     0.01    0.00        0.04               0.95




