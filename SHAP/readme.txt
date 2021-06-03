For Adaboost Classifiers or Regressors, the newest version of SHAP 0.39.0 did not support this algorithm. Therefore, the source code was modified:
As for "_tree.py" in the installation path in Anaconda 3：
e.g. D:\anaconda\Lib\site-packages\shap\explainers
After line 651, following code was added:
elif safe_isinstance(model, "sklearn.ensemble.AdaBoostRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.base_estimator_.criterion, None)
            self.tree_output = "raw_value"
After line 694, following code was added:
elif safe_isinstance(model, ["sklearn.ensemble.AdaBoostClassifier", "sklearn.ensemble._weighted_boosting.AdaBoostClassifier"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling) for e in model.estimators_]
            self.objective = objective_name_map.get(model.base_estimator_.criterion, None) #This line is done to get the decision criteria, for example gini.
            self.tree_output = "probability" #This is the last line added
For your convenience, a modified .py file was uploaded in this folder.