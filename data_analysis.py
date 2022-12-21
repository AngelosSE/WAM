from matplotlib import pyplot as plt
import plot_tools as PT
import seaborn as sns
import pandas as pd
import numpy as np
import evaluate_analysis_utilities as EAU

def count_road_users(df):
    """
    Count number of road users per location and class.
    """
    return (
                df
                .groupby(['locationId','class'])
                .apply(lambda g: len(g['objectId'].unique()))
                .unstack(level='locationId')
            )

def analyze_road_user_counts(df_raw,df_processed,path):
    n_raw = count_road_users(df_raw)
    n_processed = count_road_users(df_processed)
    print(n_raw.astype(str) + " / " + n_processed.astype(str))
    if path is not None:
        table = n_raw.astype(str) + " / " + n_processed.astype(str)
        table.to_latex(path + 'numberOfRoadUsers.tex'
            ,caption = 'Number of road users per location and class in original/processed datasets.'
            ,label= 'tab:data_inspection:numberRoadUsers')

def hist(df,attribute,units,path = None,width=PT.LatexWidth.IEEE_JOURNAL_COLUMN.value,use_legend=True):
    axs = []
    with sns.axes_style("darkgrid"): # https://stackoverflow.com/questions/29235206/changing-seaborn-style-in-subplots
        for _ in df['class'].unique():
            _,ax = plt.subplots(figsize=PT.fig_size(width))
            axs.append(ax)

    for (clss,group),ax in zip(df.groupby('class'),axs):
        sns.histplot(data =group,x=attribute,hue=group['locationId'].astype(str),
                             element = 'step',stat='percent',common_norm=False
                             ,fill=True,ax=ax,legend=use_legend)
        ax.set_xlabel(f'{attribute} [{units}]')
        ax.set_ylabel(f'percent of samples [%]')
        
    #return axs
        
        if path is not None:
            ax.get_figure().savefig(path+f"hist_of_{attribute}_class={clss}.pgf",bbox_inches="tight")
#            ax.get_figure().savefig(path+f"hist_of_{attribute}_class={clss}.png"
#                                    ,bbox_inches="tight"
#                                    ,dpi=300
#                                    )

def analyze_speeds(df,path,locationIds,width=PT.LatexWidth.IEEE_JOURNAL_COLUMN.value,use_legend=True): 
    # TODO: Add an argument classes to make it possible to plot a single class.
    # see OneNote>Research>Notes>How to formulate a plotting method in Matplotlib for figures with subplots.
    df = df[df['locationId'].isin(locationIds)]
    hist(df[(df['class']=='pedestrian') | (df['class']=='bicycle')]
            ,attribute='speed',units='meter/second', path = path, width=width
            ,use_legend=use_legend
        )

    # For vehicles samples at low speeds are so overrepresentated that a special figure is required.
    with sns.axes_style("darkgrid"): # https://stackoverflow.com/questions/29235206/changing-seaborn-style-in-subplots
        fig,axs = plt.subplots(1,2,figsize=PT.fig_size(width)
                                ,gridspec_kw={'width_ratios':[1,4],'wspace':0.5})
    group = df[df['class']=='vehicle']
    for ax in axs:
        sns.histplot(data =group,x='speed',hue=group['locationId'].astype(str),
                             element = 'step',stat='percent',common_norm=False
                             ,fill=True,ax=ax,legend=use_legend)  
    if use_legend:
        axs[0].get_legend().remove()
    axs[0].set_xlim(0,0.5)
    axs[0].set_xlabel('')
    axs[0].set_ylabel('percent of samples [%]')
    axs[1].set_xlim(0.5,axs[1].get_xlim()[1])
    axs[1].set_ylim(0,1.75)
    axs[1].set_ylabel('')
    axs[1].set_xlabel('speed [meter/second]')#,loc='left')
    axs[1].xaxis.set_label_coords(0.1,-0.2)
    #axs[1].xaxis.set_label_coords(0.1,0)
    #axs[1].xaxis.labelpad = 200
    
    axs[1].set_xticks([0.5,5,10,15])
    axs[1].set_xticklabels(['0.5','5','10','15'])
    
    if path is not None:
 #       fig.savefig(path+f"hist_of_speed_class=vehicle.png"
 #                                       ,bbox_inches="tight"
 #                                      ,dpi=300
 #                                      )
        fig.savefig(path+f"hist_of_speed_class=vehicle.pgf"
                                        ,bbox_inches="tight"
                                       )
                                       
    
def analyze_positions(df,path, locationIds, classes, showTestData = True,filename=None):
    df = df[df['locationId'].isin(locationIds)
            & df['class'].isin(classes)]

    axs = []
    for (locationId,clss), group in df.groupby(['locationId','class']):
        plt.figure()
        ax = plt.gca()
        axs.append(ax)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        PT.plot_background(locationId)
        PT.set_plot_limits(int(locationId),ax)

        if path is not None:
            plt.savefig(path+f'background_location{locationId}.png',bbox_inches="tight")
        if showTestData:
            for _,roadUser in group[group['isTrain']==0].groupby('objectId'):
                plt.plot(roadUser['xCenter'],roadUser['yCenter'],'r',linewidth=0.5,alpha = 1)
            for _,roadUser in group[group['isTrain']==1].groupby('objectId'):
                plt.plot(roadUser['xCenter'],roadUser['yCenter'],'lime',linewidth=0.5,alpha = 1)
        else:
            for _,roadUser in group.groupby('objectId'):
                plt.plot(roadUser['xCenter'],roadUser['yCenter'],'r',linewidth=0.5,alpha = 1)
        
        if path is not None:
            if filename is None:
                filename = f'locii_of_positions_location{locationId}_class={clss}.png'
            plt.savefig(path+filename,bbox_inches="tight")
            filename = None # Reset filename to allow loop to run and set name in next iteration. This is bad style I should split this method in two. One with forloop and one without
        
    return axs
    
def analyze_bicycles(cfg):
    df = pd.read_csv(cfg.storage_paths.data_split)
    df = df[(df['class']=='bicycle') & (df['locationId']==1)]
    
    _,ax = plt.subplots() # check if any outlier position
    plt_settingss = [{'facecolor':'b'},{'facecolor':'r','alpha':0.1}]
    for case,settings in zip(['train','test'],plt_settingss):
        isTrain = case == 'train'
        ax.scatter(df[df['isTrain']==isTrain]['xCenter']
                    ,df[df['isTrain']==isTrain]['yCenter']
                    ,label=case
                    ,s =4
                    ,**settings)
    ax.legend()
    
    _,ax = plt.subplots()
    _,bin_egdes = np.histogram(df[df['isTrain']==1]['speed'])
    for case in ['train','test']:
        isTrain = case == 'train'
        ax.hist(df[df['isTrain']==isTrain]['speed'],label=case,bins=bin_egdes)
    ax.legend()
    # combine these plots with plot of position versus 4.8 second FDE.
    # argue that the large error for bicycles is probably due to there being 
    # least data in this case, refer to the table