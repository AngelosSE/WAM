from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, GroupKFold
import pandas as pd
import numpy as np
from data_processing_utils import select
from definitions import StateAttribute
import models as my_models

class SklearnWrapper(BaseEstimator):
    def fit(self,X,y):
        n_data = X.shape[0]
        X = dict(zip(self.model.attributes_in_domain,
                map(np.squeeze,np.hsplit(X,X.shape[1]))))
        y = dict(zip(self.model.attributes_in_codomain,
                    map(np.squeeze,np.hsplit(y,y.shape[1]))))
        data = X | y
#        assert(len(set(map(np.size,self.model.restrictions.values())))==1) # The Data object can only be reverse-engineered if every restriction has a single value
        tmp = set(map(np.size,self.model.restrictions.values()))
        assert((tmp=={1}) | (len(tmp)==0)) # The Data object can only be reverse-engineered if every restriction has a single value
        data |= {attr:np.repeat(self.model.restrictions[attr],n_data)
                    for attr in self.model.restrictions.keys()} 
        self.model.fit(data)
        return self

    def set_params(self, **params):
        self.model.set_params_sklearn(**params)
        return super().set_params(**params)

    def predict(self,X):
        X = dict(zip(self.model.attributes_in_domain,
                    map(np.squeeze,np.hsplit(X,X.shape[1]))))
        y = self.model.predict(X)
        return np.column_stack([*y.values()])

def wrap_model(model,parameters):
    """Dynamically create a subclass of SklearnWrapper that wraps model such 
    that the subclass can interface with Sklearn. 
    
    Args:
        model:
        
        parameters: A dict whose keys are the parameter names, the values are
            irrelevant"""
    kwargs = [f'{k}={v}' for k,v in parameters.items()]
    #code = f"""def __init__(self,{','.join(kwargs)}): pass"""  # OK, for my WAM models
    code = f"""def __init__(self,{','.join(kwargs)}): 
        for  param,val in list(locals().items())[1:]:
            setattr(self,param,val)""" # For my own models I dont need setattr, but for my MLPWrapper I need it, why?
    scope = {}
    exec(code,scope)
    return type(f'Wrapped_{type(model).__name__}',(SklearnWrapper,),{'model':model,'__init__':scope['__init__']}|parameters)

def fit_manager(Wraped,data,parameters): 
    X = select(data,
        Wraped.model.restrictions,
        Wraped.model.attributes_in_domain)
    y = select(data,
            Wraped.model.restrictions,
            Wraped.model.attributes_in_codomain)
    X = np.column_stack([*X.values()]) 
    y = np.column_stack([*y.values()]) 
    sample_to_object = select(data,
                        Wraped.model.restrictions,
                        [(StateAttribute.OBJECT_ID,0)]).popitem()
    group_kfold = GroupKFold(n_splits = 5)
    CV = GridSearchCV(Wraped(), parameters
        ,scoring = 'neg_root_mean_squared_error'
        ,cv=group_kfold.get_n_splits(X,y,sample_to_object))
    CV.fit(X,y)
    result = pd.DataFrame(CV.cv_results_)
    return result

def save_CVresult(CVresult,model,path):
    """
    Saves cross-validation results. If file at path already exists, then 
    results are merged.
    """
    CVresult = CVresult.filter(regex=('param_.*|mean_test_score|std_test_score|rank_test_score'))
    if isinstance(model, my_models.WAM_over_horizon):
        CVresult = parameters_to_columns(CVresult) # to enable merging
    try:
        old_result = pd.read_csv(path)
        merged = pd.merge(old_result.drop(columns='rank_test_score')
                ,CVresult.drop(columns='rank_test_score'),how = 'outer')
        merged['rank_test_score'] = merged['mean_test_score'].rank(ascending=False)
        CVresult = merged
    except:
        pass

    CVresult.to_csv(path,index=False)

def parameters_to_columns(CVresult):
    n_parameters = CVresult['param_parameters'][0].size
    parameters = pd.DataFrame(np.row_stack(CVresult['param_parameters'].to_numpy())
                    ,columns=['param_'+str(i) for i in range(1,n_parameters+1)])
    CVresult = CVresult.drop(columns='param_parameters')
    return pd.concat([CVresult,parameters],axis = 1)