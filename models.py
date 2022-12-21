from itertools import chain
import numpy as np
from sklearn.neural_network import MLPRegressor
from numpy.typing import ArrayLike,NDArray
from data_processing_utils import select
from definitions import Attribute, StateAttribute
import time
from WAM_model import *

class time_prediction():
    lst = []
    def __call__(self,predict):
        def wrapper(wrapped_self,X):
            start = time.time()
            yhat = predict(wrapped_self,X)
            end = time.time()
            n_test = X[StateAttribute.X_CENTER,0].shape[0]
            time_prediction.lst.append((end-start,n_test))
            return yhat
        return wrapper

    @staticmethod
    def get_average_time():
        total_time = 0
        total_test = 0
        for time,n_test in time_prediction().lst:
            total_time += time
            total_test += n_test
        return total_time,total_test,total_time/total_test

def interleave(*lists):
    return list(chain(*zip(*lists)))

class Predictor_constVel():

    def __init__(
            self,
            horizons : list[int],
            restrictions : dict[Attribute,ArrayLike],
            name
            ):

        self.attributes_in_domain = \
            [ (StateAttribute.X_CENTER,0), (StateAttribute.Y_CENTER,0), 
             (StateAttribute.X_VELOCITY,0),(StateAttribute.Y_VELOCITY,0) ]

        self.attributes_in_codomain = interleave( # change to double for-loop?
                    [ (StateAttribute.X_CENTER, h) for h in horizons ],
                    [ (StateAttribute.Y_CENTER, h) for h in horizons ]
                    )
        self.restrictions = {k:np.array(v) for k,v in restrictions.items()} # wrap in array for np.isin in method restrict
        self.horizons = horizons
        self.name = name

    def fit(self,data):
        return None
    
    def predict(self,
            X : dict[Attribute,NDArray] ) -> dict[Attribute,NDArray]:
        """
        Args:
            X: An array of shape (n_data,n_attributes) whose columns,
               left to right, should have attributes self.attributes_in_domain.
               Every row of X is a value in the domain.

        Returns:
            y: A dict whose keys, for a positive integer h, consist of 
            (StateAttributte.X_CENTER,h) and (StateAttributte.Y_CENTER,h).
        
        Note: 
            Composite StateAttributes would make sense? For example, it would 
            be convenient to have StateAttribute.POSITION.
        """
        position = np.column_stack([X[StateAttribute.X_CENTER,0],
                                    X[StateAttribute.Y_CENTER,0]])
        velocity = np.column_stack([X[StateAttribute.X_VELOCITY,0],
                                    X[StateAttribute.Y_VELOCITY,0]])
        yhat = {}
        for h in self.horizons:
            future_position = position + velocity*h/2.5
            print('hardcoded sampling rate')
            yhat[(StateAttribute.X_CENTER,h)] = future_position[:,0]
            yhat[(StateAttribute.Y_CENTER,h)] = future_position[:,1]
        return yhat

class MLPWrapper(MLPRegressor):
    def __init__(self,
            attributes_in_domain,
            attributes_in_codomain,
            restrictions : dict[Attribute,ArrayLike],
            hidden_layer_sizes,
            max_iter,
            name,
            *args,
            **kwargs
            ):
        self.attributes_in_domain = attributes_in_domain
        self.attributes_in_codomain = attributes_in_codomain
        self.restrictions = {k:np.array(v) for k,v in restrictions.items()}
        self.name = name
        super().__init__(hidden_layer_sizes=hidden_layer_sizes
            ,max_iter=max_iter
            ,random_state = 0 # for reproducability
            ,*args
            ,**kwargs)

    def fit(self,data):
        X = select(data,
                self.restrictions,
                self.attributes_in_domain)
        X = np.column_stack([*X.values()])
        y = select(data,
                self.restrictions,
                self.attributes_in_codomain)
        y = np.column_stack([*y.values()])
        super().fit(X,y)

    def predict(self,X):
        X = np.column_stack([*X.values()])
        yhat = super().predict(X)
        return dict(zip(self.attributes_in_codomain,
                    map(np.squeeze,np.hsplit(yhat,yhat.shape[1]))))

    def set_params_sklearn(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes


class MyMLP(): 
    def __init__(self,attributes_in_domain, horizons, restrictions,hidden_layer_sizes,max_iter,name
            ,stateAttrs_in_codomain
            ,stateAttrs_in_codomain_displ 
            ,*args,**kwargs):

        self.attributes_in_domain = attributes_in_domain
        self.horizons = horizons
        self.restrictions = restrictions
        self.stateAttrs_in_codomain = stateAttrs_in_codomain
        self.stateAttrs_in_codomain_displ=stateAttrs_in_codomain_displ
        
        self.attributes_in_codomain = []
        for h in self.horizons:
            for stateAttr in stateAttrs_in_codomain:
                self.attributes_in_codomain.append((stateAttr,h))

        # Setup displacement predictor
        attrs_in_codomain_displ = []
        
        for h in self.horizons:
            for stateAttr in stateAttrs_in_codomain_displ:
                attrs_in_codomain_displ.append((stateAttr,h))
        
        self.MLP = MLPWrapper(attributes_in_domain
                            ,attrs_in_codomain_displ
                            ,restrictions
                            ,hidden_layer_sizes
                            ,max_iter
                            ,name
                            ,*args,**kwargs)

        self.name = name

    def fit(self,data):
        self.MLP.fit(data)
    
    def predict(self,X):
        dhat = self.MLP.predict(X)
        yhat = {}
        for h in self.horizons:
            for posStateAttr,displStateAttr in zip(self.stateAttrs_in_codomain,self.stateAttrs_in_codomain_displ): 
                yhat[posStateAttr,h] = X[posStateAttr,0] + dhat[displStateAttr,h]
#            yhat[StateAttribute.X_CENTER,h] = X[StateAttribute.X_CENTER,0]\
#                                        + dhat[StateAttribute.X_DISPLACEMENT,h]
#            yhat[StateAttribute.Y_CENTER,h] = X[StateAttribute.Y_CENTER,0]\
#                                        + dhat[StateAttribute.Y_DISPLACEMENT,h]
        return yhat

