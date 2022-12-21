from time import sleep
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydra
from evaluate_analysis_utilities import plot_background


def load_data():
    hydra.core.global_hydra.GlobalHydra.instance().clear() 
    hydra.initialize(config_path="./conf", job_name="test_app")
    cfg = hydra.compose(config_name="config")

    dtypes = {
        'recordingId':'uint',
        'locationId':'uint', 
        'orthoPxToMeter':'double', # [m/px]
        'objectId':'uint',
        'frame':'uint',
        'xCenter':'double', # [m], position relative the coordinate system of the location
        'yCenter':'double', # [m], position relative the coordinate system of the location
        'xVelocity':'double',
        'yVelocity':'double',
        'class' : 'str'
        }

    keep_every_nth = int(cfg.dataset.sampling_rate / cfg.data_processing.sampling_rate)

    # Classic approach
    #data = data.loc[data['locationId']==2]
    #data =  pd.read_csv(cfg.paths.processed_data, usecols = list(dtypes) ,dtype = dtypes)
    #downsampled_data = data.groupby('objectId').apply(lambda g: g[::keep_every_nth])

    # Composition approach
    data = \
    (
        pd.read_csv(cfg.storage_paths.processed_data, usecols = list(dtypes) ,dtype = dtypes)
        .pipe(lambda x: x.loc[x['locationId'] == 2]) 
        .pipe(lambda x: x.loc[x['class'] == 'vehicle'])
        .drop(columns=['class','locationId'])
        .groupby('objectId').apply(lambda g: g[::keep_every_nth]) # downsample
        .reset_index('objectId',drop=True) # the groupby will give objectId as both index and column
    )
    
    return data

def plot_location(data):
    #plt.close('all') # Otherwise the plot is not redrawn when backend is widget
    fig,ax = plt.subplots()
    plot_background(data.loc[data.index[0],'recordingId']
                ,data.loc[data.index[0],'orthoPxToMeter'],ax=ax)
    ax.set_ylim([-50,0])
    #ax.axline()
    ax.annotate('A',(95,-20),(110,-20), arrowprops=dict(arrowstyle="->",color='r'),color='r')
    ax.annotate('B',(32,-7),(10,-7), arrowprops=dict(arrowstyle="->",color='g'),color='g')
    #plt.show()

def select_vehicles(data):
    # Filter data for vehicles in lane A
    # Pick vehicles whose xCenter is larger than 65 and whose mean velocity has negative x-component
#    IDs = \
#    (
#        data[data['xCenter']>65] # select samples in lane of interest and an extranaeous lane
#        .groupby('objectId').filter(lambda x: x['xVelocity'].median()<0) # remove samples corresponding to vehicles moving in the wrong direction
#        ['objectId'].unique() # get ID of vehicles of interest
#    )
#    target_vehicles  = \
#    (
#        data[data['objectId'].isin(IDs)] # select full trajectory of vehicles of interest
#        .groupby('objectId').filter(lambda x: x.loc[x.index[-1],'xCenter']<65) # select only vehicles entering the intersection
#    )

    target_vehicles =\
    (
        data
        #.sort_values(['objectId','frame'])
        .groupby('objectId')
        .filter(lambda g: (g.loc[g.index[0],'xCenter']>=65)
                            & (g.loc[g.index[-1],'xCenter']<65))
    )
    IDs = target_vehicles['objectId'].unique()
    #print(3737 in IDs)
    #return target_vehicles


    fig,ax = plt.subplots()
    plot_background(data.loc[data.index[0],'recordingId']
                    ,data.loc[data.index[0],'orthoPxToMeter'],ax=ax)
    ax.set_ylim([-50,0])
    #ego_vehicles.plot('xCenter','yCenter',ax = plt.gca(),style='.r')
    #print(len(ego_vehicles['objectId'].unique()))
#    vehicles = \
#    (
#        pd.merge(target_vehicles,data[~data['objectId'].isin(IDs)]
#                ,on=['frame','recordingId'],suffixes = ['_target','_ego']) # merge ego_vehicles and non_ego_vehicles on frame and recordingId to find target vehicles
#        #.filter(regex='[a-z]*_target',axis=1)
#        .groupby('objectId_ego')
#        .filter(lambda g: g.loc[g.index[0],'yCenter_ego']>-20) # ego_vehicles originate from lane B
#    )
    #display(vehicles)

    
#    vehicles = data[ (data['recordingId'].isin(target_vehicles['recordingId'])) 
#                & (data['frame'].isin(target_vehicles['frame']))]
#    vehicles['isTarget'] = vehicles['objectId'] \
#                            .isin(target_vehicles['objectId'].unique())
    
    ego_vehicles = \
    (
        data[ (data['recordingId'].isin(target_vehicles['recordingId'])) 
                & (data['frame'].isin(target_vehicles['frame']))] # select all data coinciding with the data of the target vehicles
        .pipe(lambda vehicles:
            vehicles[~vehicles['objectId']
                        .isin(target_vehicles['objectId'].unique())]) # select candidates for ego-vehicles, by discarding all data that corresponds to target vehicles
        .groupby('objectId')
        .filter(lambda g: (g.loc[g.index[0],'yCenter']>-20)
                           & (g.loc[g.index[-1],'yCenter']<=-20)) # select ego vehicles, they originate from lane B
    )

    vehicles = pd.concat([target_vehicles,ego_vehicles],axis=0)
    vehicles['isTarget'] = vehicles['objectId'] \
                            .isin(target_vehicles['objectId'].unique())

    vehicles[vehicles['isTarget']].plot('xCenter','yCenter',style='.r',ax=ax)
    vehicles[~vehicles['isTarget']].plot('xCenter','yCenter',style='.g',ax=ax)

#    vehicles.plot('xCenter_ego','yCenter_ego',style='.g',ax = ax)
#    vehicles.plot('xCenter_target','yCenter_target',style='r.',ax=ax)
    
    critical_region = mpl.patches.Polygon(np.array([[52,-24],[37,-25],[41,-37],[53,-33]]),facecolor = 'b')
#    plt.gca().add_collection(mpl.collections.PathCollection([critical_region]))
    ax.add_patch(critical_region)
    return vehicles
    #return to_long(vehicles), target_vehicles#.sort_values(['objectId','frame'])

def select_vehicles_2(data):
    """
    Select vehicles such that for every vehicle that originates in the target 
    lane and passes enters the intersection, there exists during the same time 
    a vehicle in the ego lane that either enters the intersection before or 
    after the target vehicle.
    """
    # Filter data for vehicles in lane A, the target lane
    target_vehicle_candidates  =\
    (
        data
        #.sort_values(['objectId','frame'])
        .groupby('objectId')
        .filter(lambda g: (g.loc[g.index[0],'xCenter']>65)
                            & (g.loc[g.index[-1],'xCenter']<=65))
    ) # candidates since not guaranteed that an ego vehicle exists
    ego_vehicle_candidates = data[~data['objectId'].isin(
                            target_vehicle_candidates['objectId'].unique())]
    vehicles = \
    (
        pd.merge(target_vehicle_candidates,ego_vehicle_candidates
                ,on=['frame','recordingId'],suffixes = ['_target','_ego']) # merge ego_vehicles and non_ego_vehicles on frame and recordingId to find target vehicles
        .groupby('objectId_target')
            .filter(lambda g: (g.loc[g.index[0],'xCenter_target']>65)
                        & (g.loc[g.index[-1],'xCenter_target']<=65)) # Drop any pair of candidate target and candidate ego vehicles such that the target does not pass through the intersection
        .groupby('objectId_ego')
            .filter(lambda g: (g.loc[g.index[0],'yCenter_ego']>-25)
                        & (g.loc[g.index[-1],'yCenter_ego']<=-25))
    )
    target_vehicle_IDs = vehicles['objectId_target'].unique()
    vehicles = to_long(vehicles)
    vehicles['isTarget'] = vehicles['objectId'] .isin(target_vehicle_IDs)

    

#    vehicles.plot('xCenter_ego','yCenter_ego',style='.g',ax = ax)
#    vehicles.plot('xCenter_target','yCenter_target',style='r.',ax=ax)
    fig,ax = plt.subplots()
    plot_background(data.loc[data.index[0],'recordingId']
                    ,data.loc[data.index[0],'orthoPxToMeter'],ax=ax)
    ax.set_ylim([-50,0])
    vehicles[vehicles['isTarget']].plot('xCenter','yCenter',style='.r',ax=ax)
    vehicles[~vehicles['isTarget']].plot('xCenter','yCenter',style='.g',ax=ax)
    critical_region = mpl.patches.Polygon(np.array([[52,-24],[37,-25],[41,-37],[53,-33]]),facecolor = 'b')
#    plt.gca().add_collection(mpl.collections.PathCollection([critical_region]))
    ax.add_patch(critical_region)
    return vehicles


def is_right_of_line(position,line_vector):
        return position[0]*line_vector[1]-position[1]*line_vector[0] > 0

def test_select_vehicles():
    
    data = pd.DataFrame(
        {'recordingId': 15*[1]
        ,'frame':list(range(1,7+1)) + list(range(2,7+1)) + [6,7]
        ,'objectId':7*['target'] + 6*['ego1'] + 2*['ego2']
        ,'xCenter':list(range(-4,3,1)) + 6*[0] + [0,0]
        ,'yCenter':7*[0] + list(range(4,-2,-1)) + [4,3]})
    rectangle = np.array([[1.5,1.5],[-1.5,1.5],[-1.5,-1.5],[1.5,-1.5],[1.5,1.5]])
    fig,ax = plt.subplots()
    ax.set_aspect('equal')
    handles = {}
    handles['target'], = ax.plot(data[data['objectId']=='target']['xCenter']
                            ,data[data['objectId']=='target']['yCenter'],'.r')
    handles['ego1'], = ax.plot(data[data['objectId']=='ego1']['xCenter']
                            ,data[data['objectId']=='ego1']['yCenter'],'.g')
    handles['ego2'], = ax.plot(data[data['objectId']=='ego2']['xCenter']
                            ,data[data['objectId']=='ego2']['yCenter'],'xg')
    ax.legend(list(handles.values()),list(handles))
    ax.plot(rectangle[:,0],rectangle[:,1],'k')
    linesegments = pd.DataFrame({'xCenter': [0,-1.5,0,1.5]
                                ,'yCenter':[1.5,0,-1.5,0]
                                ,'xDirection':[-1,0,1,0]
                                ,'yDirection':[0,-1,0,1]})
    
    init_positions = \
    (
        data
        [['objectId','xCenter','yCenter']]
        .groupby(data['objectId'])
        .agg(lambda g: g.iloc[0])
    )
    diffs = {}
    for case in ['xCenter','yCenter']:
        diffs[case] = linesegments[case].to_numpy().reshape(-1,1) \
                        - init_positions[case].to_numpy().reshape(1,-1)
    closest_linesegments = dict(zip(
                            init_positions['objectId'].unique()
                            ,np.argmin(np.sqrt(diffs['xCenter']**2
                                    +diffs['yCenter']**2),axis = 0)))

    # find closest linesegment
    # filter for vehicles that is initially outside and finally passed
    init_pos_rel_closest_lineseg =\
    (
        data[['objectId','xCenter','yCenter']]
        .groupby('objectId')
        .agg(lambda g: g.iloc[0])
        .pipe(lambda df: df[['xCenter','yCenter']]
            -linesegments.iloc[[closest_linesegments[ID] for ID in df.index]][['xCenter','yCenter']].values)
    )

    final_pos_rel_closest_lineseg =\
    (
        data[['objectId','xCenter','yCenter']]
        .groupby('objectId')
        .agg(lambda g: g.iloc[-1])
        .pipe(lambda df: df[['xCenter','yCenter']]
            -linesegments.iloc[[closest_linesegments[ID] for ID in df.index]][['xCenter','yCenter']].values)
    )

    def enters_intersection(init_pos,final_pos,line_vector):
        init_is_right = is_right_of_line(init_pos,line_vector)
        final_is_left = ~is_right_of_line(final_pos,line_vector)
        return init_is_right & final_is_left

    target_vehicle_candidates  =\
    (
        data
        #.sort_values(['objectId','frame'])
        .groupby('objectId')
        .filter(lambda g: enters_intersection(init_pos_rel_closest_lineseg.loc[g['objectId'].iloc[0]].values 
            ,final_pos_rel_closest_lineseg.loc[g['objectId'].iloc[0]].values
            ,linesegments.loc[linesegments.index[1],['xDirection','yDirection']].values))
    ) # candidates since not guaranteed that an ego vehicle exists
    # I MUST SELECT FROM WHICH LANE THE VEHICLE ENTERS

    
    vehicles = \
    (
        pd.merge(target_vehicle_candidates,data,on=['recordingId','frame'],
                    suffixes = ['_target','_ego'])
        .groupby('objectId_target')
        .filter(lambda g: enters_intersection(init_pos_rel_closest_lineseg.loc[g['objectId_target'].iloc[0]].values 
            ,final_pos_rel_closest_lineseg.loc[g['objectId_target'].iloc[0]].values
            ,linesegments.loc[linesegments.index[1],['xDirection','yDirection']].values))
        .groupby('objectId_ego')
        .filter(lambda g: enters_intersection(init_pos_rel_closest_lineseg.loc[g['objectId_ego'].iloc[0]].values 
            ,final_pos_rel_closest_lineseg.loc[g['objectId_ego'].iloc[0]].values
            ,linesegments.loc[linesegments.index[0],['xDirection','yDirection']].values))
    )

    return vehicles

def to_long(vehicles):
    dfs = []
    for case in ['ego','target']:
        df = pd.DataFrame()
        df[['recordingId', 'orthoPxToMeter', 'frame', 'xCenter'
                    ,'yCenter', 'xVelocity', 'yVelocity', 'objectId']] \
                = vehicles[['recordingId', f'orthoPxToMeter_{case}'
                            , 'frame', f'xCenter_{case}'
                            , f'yCenter_{case}', f'xVelocity_{case}'
                            , f'yVelocity_{case}', f'objectId_{case}']]
        df['isTarget'] = case == 'target'
        dfs.append(df)
    return pd.concat(dfs,ignore_index=True)



def animate_data(vehicles,idx_recording):
    plt.close('all')
    # Animate the data
    fig,ax = plt.subplots()
    plot_background(vehicles['recordingId'].iloc[0]
                    ,vehicles['orthoPxToMeter'].iloc[0],ax=ax)
    ax.set_ylim([-50,0])

    # Variable number of target vehicles per ego-vehicle. How to animate all of them?
    # Many ego-vehicles can exist at in the same frame. How to select one?
    # Start with animating all all vehicles in a single recording.
    recordingIds = vehicles['recordingId'].unique()
    #frames = vehicles['frame']
    subset = vehicles[vehicles['recordingId']==recordingIds[idx_recording]]
    class animate(): 
        """
        The purpose of this class is to facilitate resetting the annotation,
        since matplotlib.axes.Axes lacks a get_annotations() method.
        It has get_children(), but it is cumbersome to filter this list for 
        the cildren of interest.
        """
        def __init__(self):
            self.annotations = []

        def __call__(self,f):
            for l in ax.get_lines():
                l.remove()
            for a in self.annotations:
                a.remove()
            self.annotations = []
            frame = subset[subset['frame']==f]
            for _,agent in frame.groupby('objectId'):
                x,y = agent['xCenter'], agent['yCenter']
                if agent.loc[agent.index[0],'isTarget']:
                    color = 'red'
                else:
                    color = 'lime'
                ax.plot(x,y,'.',color=color)
                self.annotations.append(
                    ax.annotate(agent.loc[agent.index[0],'objectId']
                                ,xy= (x,y)))

            if f == subset['frame'].iloc[-1]:
                print('Restarting')
                sleep(2)
  
    animation = mpl.animation.FuncAnimation(fig,animate()
                    ,frames=list(subset['frame']),interval=200)

    ax_button = fig.add_axes([.40,0.1,0.2,0.05])
    class Button(mpl.widgets.Button):
        def __init__(self,animation,ax,text):
            super().__init__(ax,text)
            self.animation = animation
            self.toggled = False
            super().on_clicked(self.clicked)
    
        def clicked(self,_):
            self.toggled = not self.toggled
            if self.toggled:
                self.animation.resume()
            else:
                self.animation.pause()
    button = Button(animation,ax_button,'Toggle pause')

    return animation,button



def analyze_statistics(vehicles):
    # How many target vehicles are there? 
    ntargets = len(
                    vehicles
                    [vehicles['isTarget']]
                    ['objectId']
                    .unique()
                )
    print(f'ntargets={ntargets}')

    # How many target vehicles per recording? This is interesting for 
    # understanding if applying leave-one-out is possible.
    plot_number_of_vehicles_per_recordingId(vehicles)
    # There is at least twice as many target vehicles as ego vehicles, since a 
    # target vehicle can correspond to multiple ego vehicles,.

    # How are the positions of the target and ego vehicles distributed?
    hist(vehicles,'Center')
    # The figure indicates that most target-vehicles adhere to giving right of
    # way to the ego-vehicle.

    # How are the velocities of the target and ego vehicles distributed?
    hist(vehicles,'Velocity')
    # The figure indicates that most target-vehicles adhere to giving right of
    # way to the ego-vehicle.
    # 

def plot_number_of_vehicles_per_recordingId(vehicles):
    counts = ( 
        vehicles
        .groupby(['recordingId','isTarget'])
        .apply(lambda g: len(g['objectId'].unique()))
    )
    _,ax = plt.subplots()
    labels = list(dict.fromkeys(counts.index.get_level_values('recordingId')))
    barsTrue = ax.bar(np.arange(len(labels))-0.2, counts[:,True],0.4,label='target')
    barsFalse = ax.bar(np.arange(len(labels))+0.2, counts[:,False],0.4,label ='ego')
    ax.set_xticklabels(labels)
    ax.bar_label(barsTrue)
    ax.bar_label(barsFalse)
    ax.legend()
    ax.set_xlabel('recordingId')
    ax.set_ylabel('number of vehicles')

def hist(vehicles, attr):
    fig,axs = plt.subplots(1,2)
    max_count = 0
    for isTarget in [True,False]: # to normalize figures
        h,*_ = np.histogram2d(vehicles[vehicles['isTarget']==isTarget][f'x{attr}']
                        ,vehicles[vehicles['isTarget']==isTarget][f'y{attr}'])
        max_count = max(max_count,np.max(h))
    for ax,(isTarget,name) in zip(axs,[(True,'target'),(False,'ego')]):
        plot_background(vehicles['recordingId'].iloc[0]
                    ,vehicles['orthoPxToMeter'].iloc[0],ax=ax)
        ax.hist2d(vehicles[vehicles['isTarget']==isTarget][f'x{attr}']
                ,vehicles[vehicles['isTarget']==isTarget][f'y{attr}']
                ,alpha=1,vmin = 0,vmax = max_count)
        ax.set_title(name)
        ax.set_xlabel(f'x{attr}')
        ax.set_ylabel(f'y{attr}')
        
    fig.subplots_adjust(wspace = 0.3)
    fig.suptitle('Histogram')
    cbar = plt.colorbar(mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(0,max_count))
                , orientation ='horizontal', ax=axs) 
    cbar.set_label('count')

def count_times_target_entered_first(vehicles):
    target_enters_at = 56
    ego_enters_at = -25
    # determine in which frame the target enters
    # check if the ego has already entered the intersection at that frame
    target_vehicles = vehicles[vehicles['isTarget']]
    ego_vehicles = vehicles[~vehicles['isTarget']]
    target_has_entered =  target_vehicles['xCenter'] > target_enters_at
    frames = \
    (
        target_vehicles
        [['frame','objectId']]
        [target_has_entered]
        .groupby('objectId')
        .agg(lambda g: g.iloc[0])
        ['frame']
    )
    
    vehicles = pd.merge(target_vehicles,ego_vehicles
                    ,on=['frame','recordingId'],suffixes=['_target','_ego'])
    # A problem is that I get pairs where the target vehicle is mathced with
    # an ego vehicle that has already passed the intersection.
    target_has_entered =  vehicles['xCenter_target'] < target_enters_at
    count = \
    (
        vehicles[target_has_entered] \
        .groupby('objectId_target')
        .agg(lambda g: g.iloc[0])
        .pipe(lambda x: x['objectId_ego'] > ego_enters_at)
        .sum()
    )
    return vehicles, target_has_entered,count

def plot_who_entered_first(vehicles):
    indices = (vehicles['recordingId'].isin(vehicles[~vehicles['isTarget']]['recordingId'])) \
            & (vehicles['frame'].isin(vehicles[~vehicles['isTarget']]['frame'])) # indices of eqo-vehicles
    vehicles = vehicles[indices] # select only target vehicles where an ego vehicle exists
    fig,ax = plt.subplots()
    target_enters_at = 56
    ego_enters_at = -25
    for _,g in vehicles[vehicles['isTarget']].groupby('objectId'):
        handle_target, = ax.plot(range(len(g['xCenter'])),g['xCenter']-target_enters_at,'.r')
    ax.set_ylabel('xCenter')
    #ax.axhline(56,color='red')
    ax_twin = ax.twinx()
    for _,g in vehicles[~vehicles['isTarget']].groupby('objectId'):
        handle_ego, = ax_twin.plot(range(len(g['yCenter'])),g['yCenter']-ego_enters_at,'.g')#,'og',markerfacecolor='none')
    ax_twin.set_ylabel('yCenter')
    #ax_twin.axhline(-25,color='green')
    ax_twin.axhline(0,color='black')
    ax_twin.spines['right'].set(color='green',linewidth=2)
    ax_twin.spines['left'].set(color='red',linewidth=2)
    
    ax.legend([handle_target,handle_ego],['target','ego'])
    
    ax_twin.set_ylim(*np.array([-1,1])*max(*ax_twin.get_ylim())) # align zero
    ax.set_ylim(*np.array([-1,1])*max(*ax.get_ylim()))
    ax.annotate(f'({target_enters_at})',(min(ax.get_xlim())-5,-1),fontsize='x-small', annotation_clip=False)
    ax_twin.annotate(f'({ego_enters_at})',(max(ax.get_xlim())+2.5,-0.5),fontsize='x-small', annotation_clip=False)
    

def slow_setup():
    data = load_data()

    return data

def fast_setup(data):
    plot_location(data)
    vehicles = select_vehicles(data)
    #vehicles = select_vehicles_different_lanes(data)
    
    return vehicles

def select_vehicles_different_lanes(data):
    # Filter data for vehicles in lane A
    # Pick vehicles whose xCenter is larger than 65 and whose mean velocity has negative x-component
#    IDs = \
#    (
#        data[data['xCenter']>65] # select samples in lane of interest and an extranaeous lane
#        .groupby('objectId').filter(lambda x: x['xVelocity'].median()<0) # remove samples corresponding to vehicles moving in the wrong direction
#        ['objectId'].unique() # get ID of vehicles of interest
#    )
#    target_vehicles  = \
#    (
#        data[data['objectId'].isin(IDs)] # select full trajectory of vehicles of interest
#        .groupby('objectId').filter(lambda x: x.loc[x.index[-1],'xCenter']<65) # select only vehicles entering the intersection
#    )

    target_vehicles =\
    (
        data
        #.sort_values(['objectId','frame'])
        .groupby('objectId')
        .filter(lambda g: (g.loc[g.index[0],'yCenter']<=-42)
                            & (g.loc[g.index[-1],'yCenter']>-42))
    )
    IDs = target_vehicles['objectId'].unique()
    #print(3737 in IDs)
    #return target_vehicles


    fig,ax = plt.subplots()
    plot_background(data.loc[data.index[0],'recordingId']
                    ,data.loc[data.index[0],'orthoPxToMeter'],ax=ax)
    ax.set_ylim([-50,0])
    #ego_vehicles.plot('xCenter','yCenter',ax = plt.gca(),style='.r')
    #print(len(ego_vehicles['objectId'].unique()))
#    vehicles = \
#    (
#        pd.merge(target_vehicles,data[~data['objectId'].isin(IDs)]
#                ,on=['frame','recordingId'],suffixes = ['_target','_ego']) # merge ego_vehicles and non_ego_vehicles on frame and recordingId to find target vehicles
#        #.filter(regex='[a-z]*_target',axis=1)
#        .groupby('objectId_ego')
#        .filter(lambda g: g.loc[g.index[0],'yCenter_ego']>-20) # ego_vehicles originate from lane B
#    )
    #display(vehicles)

    
#    vehicles = data[ (data['recordingId'].isin(target_vehicles['recordingId'])) 
#                & (data['frame'].isin(target_vehicles['frame']))]
#    vehicles['isTarget'] = vehicles['objectId'] \
#                            .isin(target_vehicles['objectId'].unique())
    
    ego_vehicles = \
    (
        data[ (data['recordingId'].isin(target_vehicles['recordingId'])) 
                & (data['frame'].isin(target_vehicles['frame']))] # select all data coinciding with the data of the target vehicles
        .pipe(lambda vehicles:
            vehicles[~vehicles['objectId']
                        .isin(target_vehicles['objectId'].unique())]) # select candidates for ego-vehicles, by discarding all data that corresponds to target vehicles
        .groupby('objectId')
        .filter(lambda g: (g.loc[g.index[0],'xCenter']>65)
                           & (g.loc[g.index[-1],'xCenter']<=65)) # select ego vehicles, they originate from lane B
    )

    vehicles = pd.concat([target_vehicles,ego_vehicles],axis=0)
    vehicles['isTarget'] = vehicles['objectId'] \
                            .isin(target_vehicles['objectId'].unique())

    vehicles[vehicles['isTarget']].plot('xCenter','yCenter',style='.r',ax=ax)
    vehicles[~vehicles['isTarget']].plot('xCenter','yCenter',style='.g',ax=ax)

#    vehicles.plot('xCenter_ego','yCenter_ego',style='.g',ax = ax)
#    vehicles.plot('xCenter_target','yCenter_target',style='r.',ax=ax)
    
    critical_region = mpl.patches.Polygon(np.array([[52,-24],[37,-25],[41,-37],[53,-33]]),facecolor = 'b')
#    plt.gca().add_collection(mpl.collections.PathCollection([critical_region]))
    ax.add_patch(critical_region)
    return vehicles

def filter_for_target_and_ego_at_same_time(vehicles):
    for recordingId in vehicles['recordingId'].unique():
        tmp = vehicles[vehicles['recordingId']==recordingId]
        framesTarget = tmp[tmp['isTarget']]['frame']
        framesEgo = tmp[~tmp['isTarget']]['frame']
        indices_to_drop_1 = ~framesTarget.isin(framesEgo)
        indices_to_drop_2 = ~framesEgo.isin(framesTarget)
        indices_to_drop = np.concatenate([framesTarget.index[indices_to_drop_1],framesEgo.index[indices_to_drop_2]])
        #vehicles = vehicles.drop(framesTarget.index[indices_to_drop])
        vehicles = vehicles.drop(indices_to_drop)
    return vehicles

def main():
    data = slow_setup()
    vehicles = fast_setup(data)
    vehicles = filter_for_target_and_ego_at_same_time(vehicles)
    ani,btn = animate_data(vehicles,4)
    return ani,btn

if __name__ == '__main__':
    main()
    plt.show()