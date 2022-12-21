import pickle
import numpy as np
import pandas as pd
from data_processing_utils import project, select
from definitions import Attribute, Data, StateAttribute
from numpy.typing import NDArray
import sklearn_CVwrapper
import data_processing_scout as DP_scout
import data_processing as DP
import data_analysis as DA

def load(cfg) -> Data:
    
    paths = [cfg.paths.trajectories_train,cfg.paths.trajectories_test]
    datas = {}
    for case,path in zip(['train','test'],paths):
        with open(path,'r') as file: 
            datas[case] = pickle.load(file)
    return datas

def train(models,data):

    for model in models: # Training
        model.fit(data)
    return models

def evaluate(models,data):
    yhats = []
    for model in models: # Evaluation
        X_test = select(
                    data,
                    model.restrictions,
                    model.attributes_in_domain)
        #X_test = np.column_stack([*X_test.values()])
        yhats.append(model.predict(X_test))
    return yhats #it would be more convenient, and less error prone, if predictions are stored in a dict whose keys are the model names and values are the predictions

def metric(
        y : NDArray,
        yhat : NDArray) -> NDArray :
    
    return np.linalg.norm(y-yhat,axis=1)

def assess(
        models,
        yhats : list[dict[Attribute,NDArray]], 
        data : Data
        ,stateAttrs_in_codomain = [StateAttribute.X_CENTER,StateAttribute.Y_CENTER] # TODO: Remove default value?
        ) -> list[dict[int,NDArray]]: 

    scoress = []
    for model, yhat  in zip(models,yhats): # Asses per prediction horizon
        y = select(
            data,
            model.restrictions,
            model.attributes_in_codomain)
        horizons = list(dict.fromkeys([h for _,h in model.attributes_in_codomain]))
        scores = {}
        for h in horizons:
            print(f'Assessing horizon {h}')
            attrs = [(stateAttr,h) for stateAttr in stateAttrs_in_codomain]
            y_h = project(y,attrs)
            y_h = np.column_stack([*y_h.values()])
            yhat_h = project(yhat,attrs)
            yhat_h = np.column_stack([*yhat_h.values()])
            scores[h] = metric(y_h,yhat_h)
        scoress.append(scores)
    return scoress

def scoress_to_dataframes(scoress : NDArray,
                        models, # rather, a list of pairs of model and scores
                        data,
                        DF # pandas dataframe
                        ,sampling_rate): 
    # store errors in a dataframe such that I can figure out at which positions
    # which errors occured. For this I need to know which data was used.
    # Probably select must return also the selected indices. 
    # create a pandas DataFrame with columns 
    # ['frame',objectId']+[f'err,h={h}' for h in horizons]
    # Then, pd.merge with table having all data to get the additional columns

    # attributes 'frame' and 'objectId' are keys to the sample table.
    # I need to figure out the values of 'frame' and 'objectId' of every sample
    # for which a model was evaluated.

    # The word sample has two meanings: a sample in a timeseries and a sample 
    # in the data of a model

    results_DFs = [] # better name needed.
    for model,scores in zip(models,scoress):
        sampleIds = select(
                        data,
                        model.restrictions,
                        [(StateAttribute.FRAME,0),
                            (StateAttribute.OBJECT_ID,0)])#,
                            #(StateAttribute.LOCATION_ID,0),
                            #(StateAttribute.CLASS,0)])
        results_DF = np.column_stack([*sampleIds.values()]+[*scores.values()])
        horizons = scores.keys()
        results_DF = pd.DataFrame(results_DF,
                        columns=[StateAttribute.FRAME.value,
                                StateAttribute.OBJECT_ID.value]
                                + [f'err,h={np.round(h/sampling_rate,1)}' for h in horizons])
        results_DF = pd.merge(results_DF,
                            DF,
                            on=[StateAttribute.FRAME.value,StateAttribute.OBJECT_ID.value],
                            how='left')
        results_DFs.append(results_DF)

    return results_DFs

def run_crossvalidation(model,parameters,data,path):
    Wraped = sklearn_CVwrapper.wrap_model(model
                                    ,dict.fromkeys(list(parameters.keys())))
    CVresult = sklearn_CVwrapper.fit_manager(Wraped,data,parameters)
    print(CVresult)
    sklearn_CVwrapper.save_CVresult(CVresult,model,path + model.name)

    
def run_data_processing(name
        , path
        ,sampling_rate_raw
        ,sampling_rate
        ,max_past
        ,max_horizon
        ,recordingIds = range(33)):

    if name == 'my':
        df_raw = DP.merge_into_DF(path,recordingIds)
        module = DP
    elif name == 'scout':
        df_raw = DP_scout.merge_into_DF(sampling_rate,recordingIds,path,sampling_rate_raw)
        module = DP_scout
    df_processed = module.process(df_raw, sampling_rate_raw,sampling_rate, max_past ,max_horizon)
    data = DP.extract_windows(df_processed
                                ,sampling_rate=sampling_rate
                                ,max_past=max_past
                                ,max_horizon=max_horizon)
        # in my case I also split the data.
    return df_raw,df_processed, data



def eval_and_assess(data,models,df_processed
        ,stateAttrs_in_codomain = [StateAttribute.X_CENTER,StateAttribute.Y_CENTER] # TODO: Remove default value?
        ,sampling_rate=2.5): # TODO: Remove default value?
#    n_data = list(data['train'].values())[0].size
#    indices = np.random.choice(n_data,size=int(n_data/2),replace=False)
#    data['train'] = {k:v[indices] for k,v in data['train'].items()}
#    print(f'n_data={n_data}')
    assert(~np.any(np.isin(data['train'][StateAttribute.OBJECT_ID,0],data['test'][StateAttribute.OBJECT_ID,0]))) # test and train data should not contain data from the same objects
    print('training models')
    models = train(models,data['train'])
    print('evaluating models')
    yhats = evaluate(models,data['test'])
    print('assessing_models')
    scoress = assess(models,yhats,data['test'],stateAttrs_in_codomain)
    df_processed = df_processed[['frame','objectId','locationId','class']]
    DFs = scoress_to_dataframes(scoress,models,data['test'],df_processed,sampling_rate)
    assert(~(pd.concat(DFs).isna().any().any()))
    return DFs

