import functools
import numpy as np
import pandas as pd
from definitions import StateAttribute
from plot_tools import fig_size, LatexWidth, plot_background,despine
import matplotlib.pyplot as plt
import matplotlib as mpl
import pipeline_components
import models as my_models
import data_processing
import data_processing_utils as DPU
import plot_tools as PT
from configTransaction import cfgMy as cfg
import pipeline_components as PC

def plot_for_paper_compressed(data,yhats,path,horizon=20,time=17,xlim=[35,65],ylim=[-40,-20]):
    fig = plt.figure(figsize=fig_size(LatexWidth.IEEE_JOURNAL_COLUMN.value*0.91)) # when I use full column width I my figure is wider than the column for some reason.
    ax = fig.add_axes((0,0,0.59,1)) 
    plot_background(locationId=2,ax=ax)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    #time = 17
    handles = []
    for d in [data,yhats[0],yhats[1]]:
        handles.append(*ax.plot(d[StateAttribute.X_CENTER,horizon][time]
                ,d[StateAttribute.Y_CENTER,horizon][time]))
    handles.append(*ax.plot(data[StateAttribute.X_CENTER,0][time]
                     ,data[StateAttribute.Y_CENTER,0][time]))
    handles.append(*ax.plot(data[StateAttribute.X_CENTER_EGO,0][time]
                     ,data[StateAttribute.Y_CENTER_EGO,0][time]))

    settingss = \
        [
        {'marker':'o', 'color':'lime', 'linestyle':'none','markersize':7,'markeredgewidth':2}
        ,{'marker':'x', 'color':'r', 'linestyle':'none','markersize':8,'markeredgewidth':2}
        ,{'marker':'+', 'color':'r', 'linestyle':'none','markersize':10,'markeredgewidth':2}
        ,{'marker':'o', 'color':'k', 'linestyle':'none'}
        ,{'marker':'o', 'color':'orange', 'linestyle':'none','markersize':7,'markeredgewidth':2}
        ]
    for h,settings in zip(handles,settingss):
        h.set(**settings)
    #fig.subplots_adjust(bottom=0.5)
    names = [
            r'$\bar{p}_{1,t^*+H}^\mathrm{target}$'#'ground truth'
            ,r'$\hat{p}^H$ with $\tilde{\sigma}$'#'prediction with interaction'
            ,r'$\hat{p}^H$ with $\sigma$'#'prediction without interaction'
            ,r'$p^\mathrm{target}=\bar{p}_{1,t^*}^\mathrm{target}$'#'current target position'
            ,r'$p^\mathrm{other}=\bar{p}_{1,t^*}^\mathrm{other}$'#'current ego positions'
            ]
    #fig.legend(handles,names,loc='lower center',ncol=2)
    #ax.legend(handles,names,loc='center')#,bbox_to_anchor=(0.5,0.1))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.legend(handles,names,loc='center right',borderaxespad=0.2,handletextpad=0.25,borderpad=0.1)#,loc='center',bbox_to_anchor=(0.5, -0.25))
    #fig.legend(handles,names,loc='center right',bbox_to_anchor=(1.01, 0.5))
    if path is not None:
        fig.savefig(path+"interaction_example.pgf",bbox_inches="tight")
        #fig.savefig(path+"interaction_example.png",bbox_inches="tight",dpi=300)
#    lgnd = plt.figure()
#    lgnd.legend(handles,names,loc='center')

def filter_for_target_and_ego_at_same_time(data):
    framesTarget = data[data['originalObjectId'] == 384]['frame']
    framesEgo = data[data['originalObjectId']==386]['frame']
    indices_to_drop_1 = ~framesTarget.isin(framesEgo)
    indices_to_drop_2 = ~framesEgo.isin(framesTarget)
    indices_to_drop = np.concatenate([framesTarget.index[indices_to_drop_1],framesEgo.index[indices_to_drop_2]])
    data = data.drop(indices_to_drop)
    return data

def structure_data(data, n_future):
    ego = data[data['originalObjectId']==386]
    target = data[data['originalObjectId'] == 384]
    #indices_to_keep = ego['frame'].isin(target['frame'])
    #ego = ego.loc[ego.index[indices_to_keep]]
    #indices_to_keep = target['frame'].isin(ego['frame'])
    #target = target.loc[target.index[indices_to_keep]]

    df1 = pd.DataFrame(
        {
            StateAttribute.FRAME : target['frame'].to_numpy().squeeze(),
            StateAttribute.OBJECT_ID : target['objectId'].to_numpy().squeeze(),
            StateAttribute.X_CENTER : target['xCenter'].to_numpy().squeeze(),
            StateAttribute.Y_CENTER : target['yCenter'].to_numpy().squeeze(),
            StateAttribute.X_CENTER_EGO : ego['xCenter'].to_numpy().squeeze(),
            StateAttribute.Y_CENTER_EGO : ego['yCenter'].to_numpy().squeeze(),
            StateAttribute.X_ORIENTATION : target['xOrientation'].to_numpy().squeeze(),
            StateAttribute.Y_ORIENTATION : target['yOrientation'].to_numpy().squeeze(),
            StateAttribute.SPEED : target['speed'].to_numpy().squeeze(),
            StateAttribute.ORIGINAL_OBJECT_ID: target['originalObjectId'].to_numpy().squeeze()
        })
    target = data[data['originalObjectId']==61]
    df2 = pd.DataFrame(
        {
            StateAttribute.FRAME : target['frame'].to_numpy().squeeze(),
            StateAttribute.OBJECT_ID : target['objectId'].to_numpy().squeeze(),
            StateAttribute.X_CENTER : target['xCenter'].to_numpy().squeeze(),
            StateAttribute.Y_CENTER : target['yCenter'].to_numpy().squeeze(),
            StateAttribute.X_CENTER_EGO : np.array([np.nan]*target.shape[0]), # 0
            StateAttribute.Y_CENTER_EGO : np.array([np.nan]*target.shape[0]),
            StateAttribute.X_ORIENTATION : target['xOrientation'].to_numpy().squeeze(),
            StateAttribute.Y_ORIENTATION : target['yOrientation'].to_numpy().squeeze(),
            StateAttribute.SPEED : target['speed'].to_numpy().squeeze(),
            StateAttribute.ORIGINAL_OBJECT_ID: target['originalObjectId'].to_numpy().squeeze()
        })
    df = pd.concat([df1,df2],axis=0)
    state_attrs = [StateAttribute.FRAME,
                    StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER, 
                    StateAttribute.Y_CENTER,
                    StateAttribute.X_CENTER_EGO,
                    StateAttribute.Y_CENTER_EGO,
                    StateAttribute.X_ORIENTATION,
                    StateAttribute.Y_ORIENTATION,
                    StateAttribute.SPEED,
                    StateAttribute.ORIGINAL_OBJECT_ID
                    ]
    const_state_attrs = [StateAttribute.OBJECT_ID,StateAttribute.ORIGINAL_OBJECT_ID]
    n_past = 0
    #sampling_time = 0.4
    #n_future = horizon/sampling_time
    #print(n_future)
    #assert(n_future.is_integer())
    #n_future = int(n_future)
    data = data_processing.create_data(df,state_attrs,const_state_attrs, n_past, n_future)
    
    return data

def create_models_for_real_with_velocity(horizon=20):
    attributes_in_domain = [(StateAttribute.X_CENTER,0), 
                            (StateAttribute.Y_CENTER,0),
                            (StateAttribute.SPEED,0),
                            (StateAttribute.X_ORIENTATION,0),
                            (StateAttribute.Y_ORIENTATION,0),
                            (StateAttribute.X_CENTER_EGO,0),
                            (StateAttribute.Y_CENTER_EGO,0)]
    
    param_pos = 0.5
    param_speed = 1
    param_orient = 200
    param_pos_ego = 1
    models = []
    def weights(Xbar,x,transformation,diff,**parameters):
        w = np.zeros(list(Xbar.values())[0].size)
        # if target traffic situation lacks ego vehicle,
        # then use only data where an ego vehicle is lacking
        if np.all(np.isnan(x[StateAttribute.X_CENTER_EGO,0])): # if lacks ego
            idx = np.isnan(Xbar[StateAttribute.X_CENTER_EGO,0]) # select data lacking ego
            w_nonzero = my_models.weights(
                    DPU.project(DPU.restrict_indices(Xbar,idx), attributes_in_domain[:-2])
                    ,DPU.project(x, attributes_in_domain[:-2])
                    ,transformation,diff,parameters = np.array([param_pos,param_pos,param_speed,param_orient]))   
        else: # if has ego
            idx = ~np.isnan(Xbar[StateAttribute.X_CENTER_EGO,0]) # select data having ego
            w_nonzero = my_models.weights(
                    DPU.restrict_indices(Xbar,idx)
                    ,x
                    ,transformation,diff,parameters = np.array([param_pos,param_pos,param_speed,param_pos_ego,param_pos_ego,param_orient]))
        w[idx] = w_nonzero

        return w

    def diff(Xbar,x):
        if (StateAttribute.X_CENTER_EGO,0) in x.keys():
            attrs_standard_diff = attributes_in_domain[:3] + attributes_in_domain[-2:]
        else:
            attrs_standard_diff = attributes_in_domain[:3]
        diff_std = my_models.diff(DPU.project(Xbar,attrs_standard_diff),DPU.project(x,attrs_standard_diff))
        angles = my_models.diff_curr_orientation(Xbar,x)
        return np.column_stack([diff_std,angles])

#    attrs_standard_diff = attributes_in_domain[:3] + attributes_in_domain[-2:]
#    diff = functools.partial(my_models.diff_mix
#            ,attrs_standard_diff=attrs_standard_diff)

    models.append(my_models.WAM_over_horizon_position(
        attributes_in_domain,
        horizons=[horizon],
        restrictions={},
        weights = functools.partial(weights,
                            diff = diff,
                            transformation = my_models.transf_identity),
        weights_parameters={'parameters':np.array([param_pos,param_pos,param_speed,param_pos_ego,param_pos_ego,param_orient])},
        radius=15,
        name='interaction'
        ,stateAttrs_in_codomain= [StateAttribute.X_CENTER,
                                StateAttribute.Y_CENTER]
        ,stateAttrs_codomain_displ= [StateAttribute.X_DISPLACEMENT,
                                StateAttribute.Y_DISPLACEMENT]
        ,attributes_in_balltree=((StateAttribute.X_CENTER,0),
                                    (StateAttribute.Y_CENTER,0))
        ))

    
    diff = functools.partial(my_models.diff_mix
            ,attrs_standard_diff=attributes_in_domain[:3])
    
    models.append(my_models.WAM_over_horizon_position(
        attributes_in_domain[:-2],
        horizons=[horizon],
        restrictions={},
        weights = functools.partial(my_models.weights,
                            diff = diff,
                            transformation = my_models.transf_identity),
        weights_parameters={'parameters':np.array([param_pos,param_pos,param_speed,1])},
        radius=15,
        name='noninteraction'
        ,stateAttrs_in_codomain= [StateAttribute.X_CENTER,
                                StateAttribute.Y_CENTER]
        ,stateAttrs_codomain_displ= [StateAttribute.X_DISPLACEMENT,
                                StateAttribute.Y_DISPLACEMENT]
        ,attributes_in_balltree=((StateAttribute.X_CENTER,0),
                                    (StateAttribute.Y_CENTER,0))
        ))
    return models

def plot_traffic_situations_compressed(data,path):
    # Traffic situation: target and other
    fig = plt.figure(figsize=fig_size(LatexWidth.IEEE_JOURNAL_COLUMN.value*0.8)) # when I use full column width I my figure is wider than the column for some reason.
    ax = fig.add_axes((0,0,0.775,1))
    plot_background(locationId=2,ax=ax)
    despine(ax)
    PT.set_plot_limits(locationId=2,ax=ax)

    handles = []
    labels = []
    handles.append(*ax.plot(data[data['originalObjectId']==384]['xCenter'].iloc[0]
                            ,data[data['originalObjectId']==384]['yCenter'].iloc[0]
                            ,'x'
                            ,color='black'))
    labels.append(r'$\bar{p}_{1,1}^\text{target}$')
    handles.append(*ax.plot(data[data['originalObjectId']==384]['xCenter'].iloc[1:]
                            ,data[data['originalObjectId']==384]['yCenter'].iloc[1:]
                            ,'.'
                            ,color='black'
                            ,markersize=2.5))
    labels.append(r'$\bar{p}_{1,t}^\text{target}$')
    handles.append(*ax.plot(data[data['originalObjectId']==386]['xCenter'].iloc[0]
                            ,data[data['originalObjectId']==386]['yCenter'].iloc[0]
                            ,'x'
                            ,color='orange'))
    labels.append(r'$\bar{p}_{1,1}^\text{other}$')
    handles.append(*ax.plot(data[data['originalObjectId']==386]['xCenter'].iloc[1:]
                            ,data[data['originalObjectId']==386]['yCenter'].iloc[1:]
                            ,'.'
                            ,color='orange'
                            ,markersize=2.5))
    labels.append(r'$\bar{p}_{1,t}^\text{other}$')
    fig.legend(handles,labels,loc='center right',borderaxespad=0.2
                ,handletextpad=0.25,borderpad=0.1,handlelength=1.5)#,bbox_to_anchor=(1, 0.5))
    if path is not None:
        ax.get_figure().savefig(path+f'interaction_example_real_locii_1.pgf',bbox_inches="tight")
    
    # Traffic situation: only target
    fig = plt.figure(figsize=fig_size(LatexWidth.IEEE_JOURNAL_COLUMN.value*0.8)) # when I use full column width I my figure is wider than the column for some reason.
    ax = fig.add_axes((0,0,0.775,1))
    plot_background(locationId=2,ax=ax)
    despine(ax)
    PT.set_plot_limits(locationId=2,ax=ax)

    handles = []
    labels = []
    handles.append(*ax.plot(data[data['originalObjectId']==61]['xCenter'].iloc[0]
                            ,data[data['originalObjectId']==61]['yCenter'].iloc[0]
                            ,'x'
                            ,color='black'))
    labels.append(r'$\bar{p}_{2,1}^\text{target}$')
    handles.append(*ax.plot(data[data['originalObjectId']==61]['xCenter'].iloc[1:]
                            ,data[data['originalObjectId']==61]['yCenter'].iloc[1:]
                            ,'.'
                            ,color='black'
                            ,markersize=2.5))
    labels.append(r'$\bar{p}_{2,t}^\text{target}$')
    fig.legend(handles,labels,loc='center right',borderaxespad=0.2
            ,handletextpad=0.25,borderpad=0.1,handlelength=1.5)
    if path is not None:
        ax.get_figure().savefig(path+f'interaction_example_real_locii_2.pgf',bbox_inches="tight")

def plot_distance_VS_pred_distance_compressed_legend_inside(data,yhats,path):
    horizon=10
    fig,ax = plt.subplots(figsize=(LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27*1.05,1.25))
    #_,ax = plt.subplots(figsize=(LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27,1.5))
    colors = ['black']
    IDs = [384]
    for objectId,color in zip(IDs,colors):
        indices = data[StateAttribute.ORIGINAL_OBJECT_ID,0]==objectId
        data_obj = DPU.restrict_indices(data,indices)
        REF = (data_obj[StateAttribute.X_CENTER,0][0],data_obj[StateAttribute.Y_CENTER,0][0])
        dists = np.sqrt((data_obj[StateAttribute.X_CENTER,0][:-3]-REF[0])**2 + (data_obj[StateAttribute.Y_CENTER,0][:-3]-REF[1])**2)
        dists_future = np.sqrt((data_obj[StateAttribute.X_CENTER,horizon][:-3]-REF[0])**2 + (data_obj[StateAttribute.Y_CENTER,horizon][:-3]-REF[1])**2)
        ax.plot(dists,dists_future,'.',color = color)
    indices = data[StateAttribute.ORIGINAL_OBJECT_ID,0]==384
    dists_pred = np.sqrt((yhats[1][StateAttribute.X_CENTER,horizon][indices][:-3]-REF[0])**2 + (yhats[1][StateAttribute.Y_CENTER,horizon][indices][:-3]-REF[1])**2)
    ax.plot(dists,dists_pred,'or',markerfacecolor = 'none')
    ax.set_xlabel('current distance from initial position [m]')
    #ax.set_ylabel(f'distance {horizon*0.4} s later [m]')
    ax.set_ylabel(f'distance 4 s later [m]')
    ax.legend([r'$(\|\bar{p}_{1,t^*}^\mathrm{target}-\bar{p}_{1,1}^\mathrm{target}\|,\|\bar{p}_{1,t^*+H}^\mathrm{target}-\bar{p}_{1,1}^\mathrm{target}\|)$'
                ,r'$(\|\bar{p}_{1,t^*}^\mathrm{target}-\bar{p}_{1,1}^\mathrm{target}\|,\|\hat{p}^H_{t^*}-\bar{p}_{1,1}^\mathrm{target}\|)$']
                ,loc = 'upper left'
                ,borderaxespad=0.1,handletextpad=0,borderpad=0#,labelspacing=0.1
                ,handlelength=1
                
                )
    if path is not None:
        ax.get_figure().savefig(path+f'interaction_example_distance.pgf',bbox_inches="tight")

def main_paper(path=None):
    plt.rcParams['text.usetex'] = True
    mpl.rcParams["text.latex.preamble"] = r'\usepackage{amssymb}\usepackage{amsmath}'
    mpl.rcParams.update({"pgf.preamble":r'\usepackage{amssymb}\usepackage{amsmath}'})
    _,data, _ =  PC.run_data_processing(
                                name = cfg.name
                                ,path = cfg.dataset.path
                                ,sampling_rate_raw = cfg.dataset.sampling_rate # Hz
                                ,sampling_rate = cfg.data_processing.sampling_rate # Hz
                                ,max_past = cfg.data_processing.max_past # seconds
                                ,max_horizon = cfg.data_processing.max_horizon # seconds
                                ,recordingIds=[19,24]
                                ) 
    indices_to_keep = np.full(data.shape[0],False)
    for recId,origObjId in [(19,61),(24,384),(24,386)]: # filter for hand-picked objects
        indices_to_keep += (data['recordingId'] == recId) \
                        & (data['originalObjectId'] == origObjId)
    data = data.drop(data.index[~indices_to_keep])
    data = filter_for_target_and_ego_at_same_time(data)
    plot_traffic_situations_compressed(data,path)
    data = structure_data(data,12)
    horizon = 10
    models = create_models_for_real_with_velocity(horizon)
    pipeline_components.train(models,data)
    yhats = pipeline_components.evaluate(models,data) 
    
    plot_for_paper_compressed(data,yhats,path,horizon=horizon,time=10,xlim=[35,67],ylim=[-37,-17])
    plot_distance_VS_pred_distance_compressed_legend_inside(data,yhats,path)
    plt.rcParams['text.usetex'] = False

if __name__ == '__main__':
    plt.close('all')
    plt.show(block=False)
    main_paper()
    plt.show()