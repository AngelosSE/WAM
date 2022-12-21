from pathlib import Path
import numpy as np

import pandas as pd

from data_processing import add_orientation, create_vehicle_class, validateDF
import extract_ind_traj

def augment_with_attributes(path,path_raw_data,recordingIds = range(33)):
    """
    Merge their data with raw data to also get the attributes xVelocity, 
    yVelocity, heading,class,locationId,recordingId,frame
    """
    p = Path(path)
    p_orig = Path(path_raw_data)
    dtype_theirs= {
        'frame':'uint',
        'trackId':'uint',
        'xCenter':'double', # [m], position relative the coordinate system of the location
        'yCenter':'double', # [m], position relative the coordinate system of the location
        }

    dtype_recordingMeta = {
        'recordingId':'uint',
        'locationId':'uint',
        'orthoPxToMeter':'double' # [m/px]
        }

    dtype_tracksMeta = {
        'trackId':'uint',
        'class':'str' # Car, Truck_Bus, Pedestrian, Bicycle (includes Motorcycles)
        }
   
    dtype_tracks = {
        'recordingId':'uint',
        'trackId':'uint',
        'frame':'uint',
        'xVelocity':'double',  # [m/s], velocity relative to the coordinate system of the location
        'yVelocity':'double', # [m/s],velocity relative to the coordinate system of the location
        'heading': 'double' # [deg]
        }

    dfs = []
    nobjects_prev = 0
    for i in recordingIds: # 33
        print(f'Load recording {i}')
        if i < 10:
            count = '0'+str(i)
        else:
            count = str(i)
        df_tmp = pd.read_table(p / (count+'_Trajectories.txt')
                           ,sep=' '
                           ,names=dtype_theirs.keys()
                           ,dtype=dtype_theirs
                           ,index_col=False)
        df_tmp['recordingId'] = i
        df_tmp['frameRate'] = 2.5

        recordingMeta = p_orig / (count + '_recordingMeta.csv')
        tracksMeta = p_orig / (count + '_tracksMeta.csv')
        tracks = p_orig / (count + '_tracks.csv')
        df_recordingMeta = pd.read_csv(recordingMeta,usecols=dtype_recordingMeta.keys(),dtype=dtype_recordingMeta)
        df_tracksMeta = pd.read_csv(tracksMeta,usecols =dtype_tracksMeta.keys(),dtype=dtype_tracksMeta)
        df_tracks = pd.read_csv(tracks,usecols =dtype_tracks.keys(),dtype=dtype_tracks)
        df_tmp = pd.merge(df_tracks,df_tmp,on=['recordingId','trackId','frame']) 
        df_tmp = pd.merge(df_recordingMeta, df_tmp, on=['recordingId'])
        df_tmp = pd.merge(df_tracksMeta,df_tmp, on=['trackId'])
        df_tmp['originalObjectId'] = df_tmp['trackId']
        # assign a unique objectID to every trajectory by shifting based on largest previous objectID
        df_tmp['objectId'] = df_tmp.groupby('trackId').ngroup() + nobjects_prev
        nobjects_prev += len(df_tmp['trackId'].unique())
        df_tmp = df_tmp.drop(columns='trackId')
        dfs.append(df_tmp)

    df = pd.concat(dfs,ignore_index = True)
    df = df.sort_values(['objectId','frame'])
#    validateDF(df,25,2.5)
    return df

def process(df,sampling_freq_old,sampling_freq_new,*_,**__): # The *_ and **__ are just for making my script work for both my data and scout data
    """
    Merge car and truck-buss classes into vehicle class and add attributes 
    speed, xOrientation, yOrientation. Define train/test split.
    """
    df = add_orientation(df)# The calculation of xOrientation and yOrientation assumes that no road user begins as stationary
    df = create_vehicle_class(df)
    df['speed'] = np.sqrt(df['xVelocity']**2+df['yVelocity']**2) # m/s, add speed
    
    recordingIds_test = [5,6,14,15,16,17,26,27,28,29,32]
    df['isTrain'] = 1
    df.loc[df['recordingId'].isin(recordingIds_test),'isTrain'] = 0
    validateDF(df,sampling_freq_old,sampling_freq_new)
    
    return df

merge_into_DF = extract_ind_traj.merge_into_DF 
# Include the code like this to avoid mixing my code with the other researches 
# code. And to make it easier to version my changes to the other researches code.
