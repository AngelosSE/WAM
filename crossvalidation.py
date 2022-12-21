from itertools import product
import pipeline_components as PC
from configTransaction import cfgMy, cfgScout
from definitions import StateAttribute, Location, RoadUserClass
import numpy as np
import models_transaction as Models

def main(cfg,param_search_space,case,locationId,clss):
    restrictions = {
                    (StateAttribute.LOCATION_ID,0) : [Location(locationId).value], # The .value is required due to problem with numpy. Where to put .value? The problem: https://github.com/numpy/numpy/issues/15767, 
                    (StateAttribute.CLASS,0) : [RoadUserClass(clss).value]
                    } # don't touch
    if case == 'WAM':
        initial_model = Models.WAM(
            horizons=[12]
            ,restrictions=restrictions
            ,radius=15
            ,name=f'WAM_trans_loc{locationId}_{clss}'
            ,parameters = np.array(5*[None])).Pred_WAM_displ    
    elif case == 'MLP':
        initial_model = Models.MLP(
            horizons=[12]
            ,restrictions=restrictions
            ,name=f'MLP_trans_loc{locationId}_{clss}'
            ,hidden_layer_sizes = None
            ,max_iter=400).MLP 

    _,_,data = PC.run_data_processing(
                                    cfg.dataset.name
                                    ,cfg.dataset.path
                                    ,cfg.dataset.sampling_rate
                                    ,cfg.data_processing.sampling_rate
                                    ,cfg.data_processing.max_past
                                    ,cfg.data_processing.max_horizon)
        
    PC.run_crossvalidation(initial_model
                        ,param_search_space
                        ,data['train']
                        ,'CV_')

if __name__ == '__main__':
    case = 'WAM'
    locationId = 2
    clss = 'pedestrian'
    cfg = cfgScout
    if case == 'WAM':
        lst = list(product([0.25,0.5,1,5],[1,20,50,100],[1,50,75,100,150])) 
        #lst = list(product([0.1,0.25,0.5],[50,100,150],[50,100,150]))
        #lst = list(product([0,0.1],[50],[50]))
        #lst = [p for p in lst if p != [0.25,50,50]]
        param_search_space =  [np.array(list((t[0],)+t)) for t in lst] # don't touch
        #param_search_space = [[0.1,0.1,50.0,50.0]]
        param_search_space = list(map(np.array,param_search_space)) # don't touch
        
        param_search_space = {'parameters' : param_search_space}   # don't touch
        
    elif case == 'MLP':
        param_search_space = {'hidden_layer_sizes' : [475,500,550,600]}
        
        
    main(cfg,param_search_space,case,locationId,clss)
