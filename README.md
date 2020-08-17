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
- `cambioni`: open source dataset used for rock collisions, with four classes: (hit-and-run, graze-and-merge, merging, disruption), four features (Mass, Gamma, Angle, Velocity), from 769 points.
- `iris`: toy dataset to classify types of flowers, with three classes: (Setosa, Versicolour, Virginica), four spaces (Sepal Length, Sepal Width, Petal Length, Petal Width), from 150 points.

## Optimisation
The optimisation of the different parameters is obtained using a grid search. 

## Run the code: 
- `python main.py --model decision_tree --dataloader cambioni --optim` (example for running the model decision_tree, and optimise it on the cambioni datatset)
- `python main.py --model knn --dataloader iris` (example for running the model knn without optimisation on the iris datatset)
