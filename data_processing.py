from pathlib import Path
import numpy as np
import pandas as pd
from data_processing_utils import *
from definitions import StateAttribute
from numpy.lib.stride_tricks import sliding_window_view


def add_orientation(df):
    df['xOrientation'] = df['xVelocity'] ###
    df['yOrientation'] = df['yVelocity'] ###
    df.loc[(np.abs(df['xOrientation']) < 0.01) &(np.abs(df['yOrientation']) < 0.01),['xOrientation','yOrientation']] = np.nan ##
    denominator = np.sqrt(df['xVelocity']**2+df['yVelocity']**2).to_numpy()
    df[['xOrientation','yOrientation']] = df[['xOrientation','yOrientation']].divide(denominator,axis=0)
    df[['xOrientation','yOrientation']] = df[['xOrientation','yOrientation']].fillna(method='ffill') ##
    return df

def create_vehicle_class(df):
    df.loc[df['class']=='car','class'] = 'vehicle'
    df.loc[df['class']=='truck_bus','class'] = 'vehicle'
    return df

def discard_outliers(df):
    df = df.groupby('objectId').filter(lambda g: (g['class'].iat[0] != 'pedestrian') 
                                  or (np.all(g['speed']*3.6<=15))) # too fast pedestrians
    df = df.groupby('objectId').filter(lambda g: (g['class'].iat[0] != 'bicycle') 
                                  or (np.all(g['speed']*3.6<=35))) # too fast bicycles
    df = df.groupby('objectId').filter(lambda g: np.percentile(g['speed']*3.6,95) > 0.1) # remove stationary objects
    #df = df.groupby('objectId').filter(lambda x: x.shape[0]/2.5/60 <= 20) # unusually long trajectories, hopefully stationary agents
    
    return df

def seconds_to_samples(seconds,sampling_rate):
    """
    Converts seconds to number of samples given a sampling frequency.
    """
    n_samples = seconds * sampling_rate
    if isinstance(n_samples,int):
        return n_samples
    elif n_samples.is_integer():
        return int(n_samples)
    

def discard_short_trajs(df, max_past,max_horizon,sampling_rate):
    n_past = seconds_to_samples(max_past,sampling_rate)
    n_future = seconds_to_samples(max_horizon,sampling_rate)
    shortest_len = n_past+1+n_future
    return df.groupby('objectId').filter(
        lambda x: x.shape[0] >= shortest_len
        )

def validateDF(df,sampling_freq_old,sampling_freq_new,):
    # Assert that correspondance between objectId and originalObjectId is one-to-one
    assert(all(df.groupby('objectId')['originalObjectId'].unique().apply(len)==1))

    # Assert that frame increases according to the sampling rate
    df_tmp = df.sort_values(['objectId','frame'])
    df_tmp = df_tmp.groupby('objectId')['frame'].diff()
    df_tmp = df_tmp.dropna()
    assert(np.all(np.isclose(df_tmp,sampling_freq_old/sampling_freq_new)))

    # Assert that the index labels is an enumeration from 0 to number of rows in dataframe
    assert(np.all(df.index == np.arange(np.max(df.index)+1)))

    # Assert that the number of unique objectID is equal to the maximum objectID + 1
    assert(len(df['objectId'].unique()) == df['objectId'].max()+1)

    # Assert that an objectID corresponds to a single class
    assert(np.all(df.groupby('objectId')['class'].unique().apply(len) == 1))

    # Assert that an objectID corresponds to a single locationId
    assert(np.all(df.groupby('objectId')['locationId'].unique().apply(len) == 1))

    # Assert that the set of objectID can be ordered as an enumeration from 0 to the maximum objectID + 1
    assert(np.all(np.sort(df['objectId'].unique()) == np.arange(df['objectId'].max()+1)))

    # Assert that every objectId corresponds to a single originalObjectId
    assert(all(df.groupby('objectId')['originalObjectId'].unique().apply(len)==1)) 
    
    # Assert no NaN
    assert(~np.any(df.isna()))
    
    # Assert outliers removed correctly
    maxs = df.groupby('class')['speed'].max()
    assert(maxs['bicycle'] <= 35)
    assert(maxs['pedestrian'] <= 15)
        
    # Training and Test data must come from different recordings
    assert(set(df.loc[df['isTrain']==1,'recordingId'].unique()).isdisjoint(df.loc[df['isTrain']==0,'recordingId'].unique()))
    
    print('Assertions passed')
    return

def merge_into_DF(path_to_data,recordingIds = range(33)):
    p = Path(path_to_data)

    dtype_recordingMeta = {
        'recordingId':'uint',
        'locationId':'uint',
        'frameRate':'double', 
        'orthoPxToMeter':'double' # [m/px]
        }

    dtype_tracks = {
        'recordingId':'uint',
        'trackId':'uint',
        'frame':'uint',
        'xCenter':'double', # [m], position relative the coordinate system of the location
        'yCenter':'double', # [m], position relative the coordinate system of the location
        'xVelocity':'double',  # [m/s], velocity relative to the coordinate system of the location
        'yVelocity':'double', # [m/s],velocity relative to the coordinate system of the location
        'xAcceleration':'double',
        'yAcceleration':'double',
        'heading' : 'double' # [deg]
        }

    dtype_tracksMeta = {
        'trackId':'uint',
        'class':'str' # Car, Truck_Bus, Pedestrian, Bicycle (includes Motorcycles)
        }

    dfs = []
    nobjects_prev = 0
    for i in recordingIds: # 33
        print(f'Load recording {i}')
        if i < 10:
            count = '0'+str(i)
        else:
            count = str(i)
        recordingMeta = p / (count + '_recordingMeta.csv')
        tracks = p / (count + '_tracks.csv')
        tracksMeta = p / (count + '_tracksMeta.csv')
        df_recordingMeta = pd.read_csv(recordingMeta,usecols=dtype_recordingMeta.keys(),dtype=dtype_recordingMeta)
        df_tracks = pd.read_csv(tracks,usecols =dtype_tracks.keys(),dtype=dtype_tracks)
        df_tracksMeta = pd.read_csv(tracksMeta,usecols =dtype_tracksMeta.keys(),dtype=dtype_tracksMeta)
        df_tmp = pd.merge(df_recordingMeta, df_tracks, on=['recordingId'])
        df_tmp = pd.merge(df_tracksMeta,df_tmp, on=['trackId'])
        df_tmp['originalObjectId'] = df_tmp['trackId']
        # assign a unique objectID to every trajectory by shifting based on largest previous objectID
        df_tmp['objectId'] = df_tmp['trackId'] + nobjects_prev
        nobjects_prev += len(df_tmp['trackId'].unique())
        df_tmp = df_tmp.drop(columns='trackId')
        dfs.append(df_tmp)
    df = pd.concat(dfs,ignore_index = True)
    return df
    

def downsample(df,sampling_rate_original,sampling_rate_new):
    df_downsampled = df.groupby('objectId')\
                .apply(lambda g: g.iloc[::int(sampling_rate_original/sampling_rate_new)])
    try:
        df_downsampled = df_downsampled.reset_index('objectId',drop=True) # If int(sampling_rate_raw/sampling_rate)=1, then df_downsampled lacks 'objectId' in the index
    except:
        pass
    return df_downsampled

# In process can use function composition via pandas pipe command.
def process(df,sampling_rate_original,sampling_rate_new,max_past,max_horizon): # Process data
    """
    Loads raw data and processes this to make it ready for timeseries 
    modelling at a specific sampling rate.
    After this processing data should be remain unchanged with respect to 
    choice of model. Any transformation of data after this step is defined as 
    model specific.
    """
    df['speed'] = np.sqrt(df['xVelocity']**2+df['yVelocity']**2) # m/s, add speed
    df.loc[df['speed']<0.1,['xVelocity','yVelocity']] = 0 # This should be done before add_orientation. The direction of the velocity vector jumps around when the speed is very low.
    df = add_orientation(df) # This should be done before downsampling! The calculation of xOrientation and yOrientation assumes that no road user begins as stationary
    df = downsample(df,sampling_rate_original,sampling_rate_new) 
    df= create_vehicle_class(df)
    df = discard_outliers(df)
    df = discard_short_trajs(
        df,
        max_past,
        max_horizon,
        sampling_rate_new
        )
    df['objectId'] = df.groupby('objectId').ngroup() # Re-assign objectID due to discarding objectIDs
    df = df.sort_values(['objectId','frame']) 
    df = df.reset_index(drop=True)
    df = split_data(df)
    validateDF(df,sampling_rate_original,sampling_rate_new)
    
    return df

## SPLIT DATA

def permutations(n_digits):
    permutations = [[0],[1]]
    for i in range(n_digits-1):
        new_perms = []
        for ii in [0,1]:
            branch = [perm + [ii] for perm in permutations]
            new_perms = new_perms + branch
        permutations = new_perms
    return permutations

def split_data(df):
    print('splitting data')
    # recordingId, objectId,class
    df['isTrain'] = 0
    df_objects = df.groupby('objectId').apply(lambda x: x.head()) # discard unnessesary rows, now objectId is a key
    for locationId, group in df_objects.groupby('locationId'):
        recordingIds = group['recordingId'].unique()
        n_recordings = len(recordingIds)
        boolean_indices = permutations(n_recordings)
        portions = []  
        for isTrain in boolean_indices:
            train_recordingIds = recordingIds[np.array(isTrain,dtype=bool)]
            tmp = group[group['recordingId'].isin(train_recordingIds)]
            n_train = tmp.groupby('class').size()
            n_total = group.groupby('class').size()
            portion_train = n_train / n_total
            portions.append(list(portion_train))
        portions = pd.DataFrame(portions,columns=portion_train.index)
        errs = np.abs(portions - 0.7)
        errs = errs.dropna().sum(axis=1)
        idx = errs.index[errs.argmin()]
        print(portions.loc[idx])
        isTrain = boolean_indices[idx]
        train_recordingIds = recordingIds[np.array(isTrain,dtype=bool)]
        print(train_recordingIds)
        df.loc[df['recordingId'].isin(train_recordingIds),'isTrain'] = 1
    
    return df

#########

def extract_windows(df,sampling_rate,max_past,max_horizon
        ,state_attributes = [StateAttribute.FRAME,
                            StateAttribute.OBJECT_ID, # FRAME paired with OBJECT_ID is a key to the Samples DataFrame
                            StateAttribute.X_CENTER,
                            StateAttribute.Y_CENTER,
                            StateAttribute.X_VELOCITY,
                            StateAttribute.Y_VELOCITY,
                            StateAttribute.HEADING,
                            StateAttribute.SPEED,
                            StateAttribute.X_ORIENTATION,
                            StateAttribute.Y_ORIENTATION,
                            StateAttribute.LOCATION_ID,
                            StateAttribute.CLASS]
        ,const_state_attrs = [StateAttribute.OBJECT_ID,
                                StateAttribute.LOCATION_ID,
                                StateAttribute.CLASS]
        ,do_add_displacements=True
        ):
    max_npast = int(max_past * sampling_rate)
    max_nfuture = int(max_horizon * sampling_rate)

    data = {}
    for case in ['train','test']:
        isTrain = case == 'train'
        df_case = df[df['isTrain']==isTrain]
        data[case] = create_data(df_case,
                                    state_attributes,
                                    const_state_attrs,
                                    max_npast,
                                    max_nfuture,
                                    do_add_displacements=do_add_displacements)
    return data
    

def create_data(df,state_attrs,const_state_attrs, n_past, n_future,do_add_displacements=True)->Data:
    """
    Assumes that every trajectory in df has length larger than or equal to 
    n_past+1+n_future. Otherwise sliding_windows_view throws exception.
    Assumes that every trajectory is sorted by frame
    """
    ary = df[state_attrs].to_numpy().copy() # dtype is object, since data frame contains strings
    width = n_past+1+n_future
    n_state_attrs = len(state_attrs)
    windows = sliding_window_view(ary,(width,n_state_attrs))
    n_data = windows.shape[0]
    windows = windows.reshape(n_data,-1)
    attrs_in_windows = []
    for t in range(-n_past,n_future+1):
        for stateAttr in state_attrs:
            attrs_in_windows.append((stateAttr,t))
    columns = np.split(windows,windows.shape[1],axis=1)
    columns = map(np.squeeze,columns) # to avoid unintended broadcasting.
    data = dict(zip(attrs_in_windows,columns))

    # Some juggeling to remove unnecessary array views before copying 
    # viewed memory is required 
    #>
    const_state_attrs_wout_objId = [stateAttr for stateAttr
                                    in const_state_attrs
                                    if stateAttr is not StateAttribute.OBJECT_ID]
    const_attrs_wout_objId = list(zip(const_state_attrs_wout_objId,
                                    len(const_state_attrs_wout_objId)*[0]))
    non_const_attrs_with_objId = [(stateAttr,t) for stateAttr,t 
                                in attrs_in_windows
                                if stateAttr not in const_attrs_wout_objId]
    data = project(data,non_const_attrs_with_objId+const_attrs_wout_objId)
    assert(np.shares_memory(ary,data[[*data.keys()][0]])) 
    #<
    data = discard_illegal_data(data) # requires copying data.
    #print(np.shares_memory(ary,data[[*data.keys()][0]])) # prints false
    non_const_attrs = [(stateAttr,t) for stateAttr,t 
                        in attrs_in_windows
                        if stateAttr not in const_state_attrs]
    const_attrs = const_attrs_wout_objId \
                    + [(StateAttribute.OBJECT_ID,0)]
    data = project(data, non_const_attrs+const_attrs)
    
    cast_dtype(data) 
    if do_add_displacements:
        add_displacement(data,n_future)
    return data

def discard_illegal_data(data):
    """
    A sample is illegal if in the sample there are two time instances
    having different objectId.

    .. todo:
        To support an interaction model it may be better to use a state 
        attribute called TRAJECTORY_ID instead of OBJECT_ID, since for such 
        models a trajectory contains multiple objects.
    """
    # this is actually a restriction where values of projections on more than 
    # one attribute is considered. In my restrict method I only consider
    # restrictions where a restriction is formulated as the values of a 
    # projection on a single attribute is within some set.
    # Boolean indexing requires copying.
    objectIds = np.column_stack([vals for (stateAttr,t),vals in data.items()
                                if stateAttr is StateAttribute.OBJECT_ID])
    admissible_samples = np.all(np.diff(objectIds,axis=1) == 0, axis=1)
    return{k:v[admissible_samples] for k,v in data.items()} 

def cast_dtype(data):
    """The conversion from a Pandas DataFrame, .to_numpy(), returns an array 
    with dtype object, since the DataFrame has columns with string values. 
    Some attributes require numeric dtype, since arrays with dtype object lack
    numerical methods like sum and prouct.
    
    An alternative solution is to define the DataFrame such that any string
    is replaced by a unique numerical tag. 
    """
    
    for attr,values in data.items():
        if np.all([isinstance(v,int) for v in values]):
            data[attr] = values.astype(int)
        elif np.all([isinstance(v,float) for v in values]):
            data[attr] = values.astype(float)

def add_displacement(data,n_future):
    """
    ..todo::
    I should call the displacement states
    (StateAttributes.X_DISPLACEMENT,(0,t)) as I did in  my models
    """
    for t in range(1,n_future+1): # Requires storing a lot in memory, perhaps it is better to Predictor_WAM compute displacement when needed?
            data[StateAttribute.X_DISPLACEMENT,t] = \
                data[StateAttribute.X_CENTER,t]-data[StateAttribute.X_CENTER,0]
            data[StateAttribute.Y_DISPLACEMENT,t] = \
                data[StateAttribute.Y_CENTER,t]-data[StateAttribute.Y_CENTER,0]

