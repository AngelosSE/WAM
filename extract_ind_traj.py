# -*- coding: utf-8 -*-
"""
Created on Sun May  3 13:10:36 2020
This is the function to plot the trajectories from InD dataset
https://www.ind-dataset.com/
@author: cheng
"""

import numpy as np
import os
import pandas as pd
import extract_ind_traj_original


def merge_into_DF(new_framerate=2.5
                ,recordingIds=range(33)
                ,path = ... #'my_path/inD-dataset-v1.0/data'
                ,ori_framerate = 25):
    
    nobjects_prev = 0
    trajs = []
    for recordingId in recordingIds:
        print(f'Loading recording {recordingId}')
        tracks = pd.read_csv(os.path.join(path, f"{recordingId:02}_tracks.csv"))
        recordingMeta = pd.read_csv(os.path.join(path, f"{recordingId:02}_recordingMeta.csv"))
        tracksMeta = pd.read_csv(os.path.join(path, f"{recordingId:02}_tracksMeta.csv"))
        tracks = pd.merge(tracks,tracksMeta[['recordingId','trackId','class']])
        tracks = pd.merge(tracks,recordingMeta[['recordingId','locationId']])

        dowmsample_rate = ori_framerate / new_framerate
        
# =============================================================================
#         # Down sample the tracks from the original frame rate t0 2.5 fps (timestep=0.4s)
#         if dowmsampling:                   
#             tracks = tracks.loc[tracks['frame']%dowmsample_rate==0]
#         
#         # # Plot the trajectories on the background image
#         orthoPxToMeter = recordingMeta['orthoPxToMeter'].values[0]*12        
#         plot(tracks, bgimage_dirs[i], orthoPxToMeter, tracksMeta, i)
# =============================================================================

        seq_index = get_seq_index(tracksMeta, ori_framerate)
        traj = extract_data(tracks, seq_index, dowmsample_rate)
        traj['recordingId'] = recordingId
        traj = traj.rename({'trackId':'originalObjectId'},axis='columns')
        #traj['objectId'] = traj['originalObjectId'] + nobjects_prev_old # this fails because seq_index removes some objects and this assumes that traj['originalObjectId'] is a range of numbers
        traj['objectId'] = 0
        for i,id in enumerate(traj['originalObjectId'].unique()):
            traj.loc[traj['originalObjectId']==id,'objectId'] = i + nobjects_prev
        nobjects_prev += len(traj['originalObjectId'].unique())
        trajs.append(traj)
    trajs = pd.concat(trajs).reset_index(drop=True)
    return trajs
    
def get_seq_index(tracksMeta, ori_framerate, seq_length=8):
    '''
    This is the function to extract sequences have the minimum predefined length
    Like the trajnet, every road user will be extracted once
    It tries to keep all the trajectories in a scenario
    '''
    seq_index = []
    seq_length = seq_length*ori_framerate
    start_end = tracksMeta[['trackId', 'initialFrame', 'finalFrame']].values
    
    ini_start = np.min(start_end[:, 1])
    ini_end = ini_start+seq_length
        
    
    while not tracksMeta.empty:
        # ini_start = np.min(start_end[:, 1])
        # ini_end = ini_start+seq_length
        _tracksMeta = tracksMeta.loc[(tracksMeta['initialFrame']<=ini_start) 
                                & (tracksMeta['finalFrame']>=ini_end-1)]
        
        for i in _tracksMeta.trackId.unique():
            seq_index.append([i, ini_start, ini_end])
         
        # update the tracksMeta and remove the brocken ones    
        tracksMeta = tracksMeta.loc[tracksMeta['finalFrame']>=ini_end-1]
        tracksMeta = tracksMeta.drop(_tracksMeta.index.values)
        
        # update the initial start and end
        # ini_start = tracksMeta.initialFrame.min()
        # ini_start += seq_length/2 
        ini_start = ini_end
        ini_end = ini_start+seq_length
    
    seq_index = np.asarray(seq_index)    
    # print(seq_index)
    # print(len(seq_index))
    
    return seq_index
        
            
def extract_data(tracks, seq_index, dowmsample_rate=None):
    
    traj = []
    
    if dowmsample_rate:
        tracks = tracks.loc[tracks['frame']%dowmsample_rate==0]
    
    for index in seq_index:
        user_tracks = tracks.loc[tracks['trackId']==index[0]]
        user_tracks = user_tracks.loc[(user_tracks['frame']>=index[1])
                                      & (user_tracks['frame']<index[2])]
        traj.append(user_tracks)
    return pd.concat(traj,ignore_index = True)
    
RECORDING_ID = {1:range(7,18)
                ,2:range(18,30)
                ,3:range(30,33)
                ,4:range(7)
                }


def test_seq_length():
    df1 = merge_into_DF(new_framerate = 25, recordingIds = range(2))
    df2 = merge_into_DF(new_framerate = 2.5, recordingIds = range(2))
    print(df1.groupby('objectId')
                .apply(lambda g: g.iloc[::10])
                .reset_index(drop=True)
                .equals(df2))

def test_same_result_as_original():
    trajs = extract_ind_traj_original.main()
    df = merge_into_DF(new_framerate = 2.5)
    print(np.all(np.isclose(trajs,df[['frame','originalObjectId','xCenter','yCenter']].to_numpy())))


if __name__ == "__main__":
    #df = merge_into_DF(new_framerate = 2.5)
    #df = merge_into_DF(new_framerate = 25, recordingIds = [22,23])#RECORDING_ID[2])
    #print(df)
    #originalObjectId, recordingId = [(561,22),(0,23)]
    #test_seq_length()
    test_same_result_as_original()