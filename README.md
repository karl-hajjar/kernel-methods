# Kernel Methods for Machine learning
This repository contains the code of a project we did for our Master's class Kernel Methods for Machine Learning. The aim was
learn to learn how to implement machine learning algorithms using kernel methods, gain understanding about them and adapt them
to structural data. The most important rule was <b>do it yourself</b> : we were not allowed to use sckit-learn and had to 
re-implement classical algorithms such as SVM or Kernel Ridge Regression ourselves from sratch, being able to use only Python
librairies for optimization. 

The project was to predict decease from genetic data. We used different kernels to predict decease but it turned out in the end
the Mismatch Kernel (which we coded ourselves) coupled with a SVM gave the best result, <b>0.8</b> on the hidden part of the test set.

The <i>kernel_presentation</i> notebook gives an overview of the algotithms we coded and how they compare to sklearn, showing 
visualizations of the learning power of our algorithms on toy examples (data we generated ourselves). The rest of the Python 
files contain the code for the project : 
- <i>kernels.py</i> contains the different functions computing kernels (polynomial, Gaussian, Mismatch)
- <i>learning_methods.py</i> contains the different learning algorithms we coded (SVM, KRR)
- <i>tools.py</i> contains a list of useful functions that come in handy for training and cross validating (we re-implemented a 
  specific function train_test_split adatpted to kernels)
 - <i>main.py</i> contains a script which is run by the bash file start.sh and which trains models on the training data and predicts
   labels for the submission data (there are 3 different files to train and predict)
   
 
