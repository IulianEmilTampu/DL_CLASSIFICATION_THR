'''
Script that given a list of models, uses the summary test file to plot the ROC,
the PP comparing the models. The training performance is also plotted to show
how the different models trained (looking for overfitting).

Steps
1 - get models, models' paths. Check that all the models have the test summary file.
2 - load the values needed to plot the ROCs for comparison
3 - loop through all the models and for each get the training curves (tr and val
    loss, accuracy and F1-score) for each fold.
4 - plot overall curves (one graph for each parameter).
'''