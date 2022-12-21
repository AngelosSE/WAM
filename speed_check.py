import numpy as np
from definitions import StateAttribute
import models_WAM
from configTransaction import cfgMy
import pipeline_components as PC
import functools
import models
import time

attributes_in_domain = [(StateAttribute.X_CENTER,0), 
            (StateAttribute.Y_CENTER,0),
            (StateAttribute.SPEED,0),
            (StateAttribute.X_ORIENTATION,0), 
            (StateAttribute.Y_ORIENTATION,0)]
attrs_standard_diff = attributes_in_domain[:3]
diff = functools.partial(models.diff_mix
    ,attrs_standard_diff=attrs_standard_diff)
weights = functools.partial(models.weights,
                    diff = diff,
                    transformation = models.transf_identity)
horizons = range(1,12+1)
parameters = [0.5,0.5,1.0,50.0]
stateAttrs_in_codomain = [StateAttribute.X_CENTER,
                            StateAttribute.Y_CENTER]
attributes_in_balltree = [(StateAttribute.X_CENTER,0)
                            ,(StateAttribute.Y_CENTER,0)]

attributes_in_codomain = []
for h in horizons:
    for stateAttr in stateAttrs_in_codomain:
        attributes_in_codomain.append((stateAttr,h))

model = models.Predictor_WAM(
        attributes_in_domain = attributes_in_domain
        ,restrictions ={}
        ,weights=weights
        ,weights_parameters={'parameters':np.array(parameters)}#np.array([0.45,0.45,45,157])},
        ,radius = 15
        ,attributes_in_codomain = attributes_in_codomain
        ,attributes_in_balltree=attributes_in_balltree
        ,name='')

df_raw,df_processed, data =  PC.run_data_processing(
                            name = cfgMy.dataset.name
                            ,path = cfgMy.dataset.path
                            ,sampling_rate_raw = cfgMy.dataset.sampling_rate # Hz
                            ,sampling_rate = cfgMy.data_processing.sampling_rate # Hz
                            ,max_past = cfgMy.data_processing.max_past # seconds
                            ,max_horizon = cfgMy.data_processing.max_horizon # seconds
                            ,use_old_split = False
                            ,recordingIds=range(5))

model.fit(data['train'])
t0 = time.time()
model.predict(data['test'])
t1 = time.time()
print(t1-t0)

modelNew = models.WAM_over_horizon(
     attributes_in_domain
        , horizons
        ,restrictions ={}
        ,weights=weights
        ,weights_parameters={'parameters':np.array(parameters)}#np.array([0.45,0.45,45,157])},
        ,radius = 15
        ,stateAttrs_in_codomain=stateAttrs_in_codomain
        ,attributes_in_balltree=attributes_in_balltree)

modelNew.fit(data['train'])
t0 = time.time()
modelNew.predict(data['test'])
t1 = time.time()
print(t1-t0)

def fit():
    model.fit(data['train'])
def fitNew():
    modelNew.fit(data['train'])

def predict():
    model.predict(data['test'])
def predictNew():
    modelNew.predict(data['test'])
