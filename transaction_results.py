import os
import data_processing as DP 
import data_processing_utils as DPU 
from definitions import StateAttribute # available via appended path
import matplotlib.pyplot as plt
import numpy as np
from configTransaction import cfgMy, cfgScout
import evaluate_analysis_utilities as EAU
import data_analysis as DA
import example_explaining_large_parameters
import example_of_modelling_interactions
import cache_tools as CT
import plot_tools as PT
import pipeline_components as PC
import models
import time
import warnings


def run_data_analysis(df_raw,df_processed,paths):
    df_raw = DP.create_vehicle_class(df_raw)
    DA.analyze_road_user_counts(df_raw,df_processed,paths.save_tables)
    DA.analyze_positions(df_processed
                        , paths.save_figures
                        ,locationIds = [2]
                        ,classes=['vehicle','bicycle','pedestrian']
                        ,showTestData=False)

def analysis_FDE_smooth(DFs_merged,path):
    for clss in ['vehicle','bicycle','pedestrian']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = EAU.plot_position_VS_error_smooth_comparison(
                                            [DFs_merged['MLP'],DFs_merged['WAM']]
                                            ,clss
                                            ,locationId=2
                                            ,n_contours=6
                                            ,fig_width = PT.LatexWidth.IEEE_JOURNAL_TEXT.value/72.27*0.8)
        if path is not None:
            fig.savefig(path+f'position_VS_FDE_smooth_{clss}.png'
                                ,bbox_inches="tight"
                                ,dpi=300
                                )

def analysis(models,assessments,path,df_processed):
    model_names = {'CV':'Const. vel.','WAM':'WAM','MLP':'Neural net.'} # config
    DFs = EAU.concatenate_assessments(models,assessments)
    plt.rcParams['text.usetex'] = True
    EAU.boxplot_horizon_VS_err(
            list(DFs.values())
            ,list(model_names.values())
            ,locationIds=[1,2]
            ,classes= ['vehicle','bicycle','pedestrian']
            ,path=path
            ,figsize=(PT.LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27,1.25)
            )

    DFs_merged = EAU.merge_to_get_xCenter_yCenter(DFs,df_processed)
    analysis_FDE_smooth(DFs_merged,path)
    plt.rcParams['text.usetex'] = False

def analysis_scout(models, assessments,path):
    model_names = ['Const. vel.' ,'WAM' ,'Neural net.']
    DFs = EAU.concatenate_assessments(models,assessments)
    print('aggregate ADE/FDE')
    aggregateResults = EAU.ADE_and_FDE_statistic_per_class_location_model(
        list(DFs.values())
        ,model_names
        ,statistic = 'mean'
        ,groups = ['locationId'])
    print(aggregateResults)
#    EAU.sanity_check_ADE_FDE_calculations(DFs,model_names)

def results_on_my_data():
    df_raw,df_processed, data =  PC.run_data_processing(
                                name = cfgMy.name
                                ,path = cfgMy.dataset.path
                                ,sampling_rate_raw = cfgMy.dataset.sampling_rate # Hz
                                ,sampling_rate = cfgMy.data_processing.sampling_rate # Hz
                                ,max_past = cfgMy.data_processing.max_past # seconds
                                ,max_horizon = cfgMy.data_processing.max_horizon # seconds
                                ) 

    run_data_analysis(df_raw,df_processed,cfgMy.paths)

    df_processed = df_processed[df_processed['locationId'].isin([1,2])] 
    for case in ['train','test']:
        data[case] = DPU.restrict(data[case],{(StateAttribute.LOCATION_ID,0):np.array([1,2])})


    assessments = PC.eval_and_assess(data,cfgMy.models,df_processed,sampling_rate=cfgMy.data_processing.sampling_rate)
    if (cfgMy.paths.cache is None) | ((time.time()-os.stat(cfgMy.paths.cache+'eval_and_assess_1').st_mtime) < 2):
        print('WAM execution time:')
        print(models.time_prediction.get_average_time()) # You can only get this result without loading eval_and_assess_1 from cache.

    analysis(cfgMy.models,assessments,cfgMy.paths.save_figures,df_processed)
    example_of_modelling_interactions.main_paper(path=cfgMy.paths.save_figures)
    with warnings.catch_warnings(): 
        warnings.simplefilter("ignore")
        example_explaining_large_parameters.main_result(path=cfgMy.paths.save_figures)

def results_on_scout_data():
    ## table comparison with state of the art
    ## table mADE/mFDE
    _,df_processed, data = PC.run_data_processing(
                                name=cfgScout.name
                                ,path=cfgScout.dataset.path
                                ,sampling_rate_raw = cfgScout.dataset.sampling_rate # Hz
                                ,sampling_rate = cfgScout.data_processing.sampling_rate # Hz
                                ,max_past = cfgScout.data_processing.max_past # seconds
                                ,max_horizon = cfgScout.data_processing.max_horizon) # seconds


    assessments = PC.eval_and_assess(data,cfgScout.models,df_processed,sampling_rate=cfgScout.data_processing.sampling_rate)
    analysis_scout(cfgScout.models,assessments,cfgScout.paths.save_tables)

def skip(fun):
    def wrapper(*args,**kwargs):
        return None
    return wrapper

def setup_skipping():
    methods_to_skip = [
        (DA,'analyze_road_user_counts')
        ,(DA,'analyze_positions')
        ,(EAU,'boxplot_horizon_VS_err')
        ,(None,'analysis_FDE_smooth')
        ,(example_of_modelling_interactions,'main_paper')
        ,(example_explaining_large_parameters,'main_result')
        ]
    for module, name in methods_to_skip:
        if module is None:
            method = globals()[name]
            globals()[name] = skip(method)
        else:
            method = getattr(module, name)
            setattr(module,name,skip(method))

def main():

#    setup_skipping()

    if cfgMy.paths.cache is not None:
        PC.eval_and_assess = CT.cache_calls_on_disk(cfgMy.paths.cache)(PC.eval_and_assess)
        PC.run_data_processing = CT.cache_calls_on_disk(cfgMy.paths.cache)(PC.run_data_processing)

    results_on_my_data()
    results_on_scout_data()


if __name__ == '__main__':
    main()
    plt.show()


