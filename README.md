# modelling_week_2019

Credit Card Fraud Detection problem for the [XIII Modelling Week](http://www.mat.ucm.es/congresos/mweek/XIII_Modelling_Week/),
held in the Faculty of Mathematics of the Universidad Complutense de Madrid (UCM), during  10-14 June 2019.
The Modelling Week is open to the students of the Master in Mathematical Engineering at UCM, as well as to participants from other mathematically oriented master programs worldwide.
The purpose is to teach and guide the students to solve a realistic industry problem.

* Problem in [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
* [Link to the data](https://github.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/raw/master/creditcard.csv)

The problem can be approached in three ways: supervised, unsupervised and mixed. 
We are going to start using a supervised approach, since it is simpler. 
If time permits, we'll explore unsupervised methods (a really interesting field).

## Python libraries
`jupyter`,`pandas`,`matplotlib`,`seaborn`,`sklearn`,`tensorflow`,`keras`,`imblearn`,`xgboost`

## Outline
* Basic programming with python and `jupyter`
* Exploratory data analysis, cleaning and preprocessing. Feature engineering.
* Overfitting. Validation scheme. Difference between train, validation and test sets.
* Metrics: precision, recall, ROC curve, AUC (ROC), F1, confusion matrix. Focus on unbalanced datasets.
* Classification algorithms in `sklearn`. Comments on hyperparameter tuning. 
* `xgboost` in Python using xgboost.sklearn API.
* Combination of models. Calibration. Ensembling and Stacking.
* Neural Networks in `keras`:
    * Feed Forward Neural Network for classification.
    * Autoencoder as an anomaly detector (semi and unsupervised)
    * Autoencoder as a feature builder (unsupervised)
* Combination of unsupervised and supervised methods.

## Cheatsheets
* Jupyter ([Datacamp](https://datacamp-community-prod.s3.amazonaws.com/48093c40-5303-45f4-bbf9-0c96c0133c40))
* Numpy ([Datacamp](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf))
* Pandas ([Datacamp](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf))
* Scikit-learn ([Datacamp](https://datacamp-community-prod.s3.amazonaws.com/5433fa18-9f43-44cc-b228-74672efcd116)) 
* Matplotlib ([Datacamp](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Matplotlib_Cheat_Sheet.pdf))
* Seaborn ([Datacamp](https://datacamp-community-prod.s3.amazonaws.com/f9f06e72-519a-4722-9912-b5de742dbac4))
  
## Resources
  * Using `xgboost` in Python ([Datacamp](https://www.datacamp.com/community/tutorials/xgboost-in-python))
  * Combine `xgboost` and `sklearn` ([GitHub](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/sklearn_examples.py))
  * [Videos and slides ISLR](https://www.r-bloggers.com/in-depth-introduction-to-machine-learning-in-15-hours-of-expert-videos/)
  * [Machine Learning cheatsheets](https://stanford.edu/~shervine/teaching/cs-229/)
  * 8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset ([machinelearningmastery](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/))
  * Comparison of the different over-sampling algorithms ([imblearn](https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html#sphx-glr-auto-examples-over-sampling-plot-comparison-over-sampling-py))
  * Comparison of the different under-sampling algorithms ([imblearn](https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/under-sampling/plot_comparison_under_sampling.html#sphx-glr-auto-examples-under-sampling-plot-comparison-under-sampling-py))

## Bibliography
*  **Leo Breiman "Statistical Modeling: The Two Cultures" (2001)** ([Breiman](http://www.stat.cmu.edu/~ryantibs/journalclub/breiman_2001.pdf))
 * Elements of Statistical Learning ([ESL](https://web.stanford.edu/~hastie/ElemStatLearn/))
 * Introduction to Statistical Learning with R ([ISLR](http://www-bcf.usc.edu/~gareth/ISL/))
 * Pattern Recognition and Machine Learning ([Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf))
 * Bayesian Data Analysis ([BDA](http://www.stat.columbia.edu/~gelman/book/))