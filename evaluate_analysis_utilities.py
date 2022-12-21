import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from definitions import StateAttribute
import plot_tools as PT
import numpy as np
import models as my_models
import sklearn

def concatenate_assessments(models,assessments):
    """
    To each model corresponds a dataframe containing the assessment. A model 
    is of a certain family of relations, e.g. WAM. The domain of a model is 
    defined by restrictions. This method concatenates the assessments 
    corresponding to a family of relations.  
    """
    locationIds = np.unique(np.concatenate([m.restrictions[StateAttribute.LOCATION_ID,0] for m in models]))
    #klass = {model.__class__ for model in models} # use list(dict.fromkeys) instead to keep ordering?
    model_classes = [my_models.Predictor_constVel,my_models.WAM_over_horizon_position,my_models.MyMLP]
    names = ['CV','WAM','MLP']
    DFs = {}
    for klass,name in zip(model_classes,names):
        dfs = []
        for model,df in zip(models,assessments):
            if isinstance(model,klass)\
                    & np.all(np.isin(model.restrictions.get((StateAttribute.LOCATION_ID,0)),locationIds)) \
                    & np.all(np.isin(model.restrictions.get((StateAttribute.CLASS,0)),['vehicle','bicycle','pedestrian'])): #cfg.experiment.restrictions.locations:     #cfg.experiment.restrictions.classes:
                dfs.append(df)
        DFs[name] = pd.concat(dfs,axis=0)
    return DFs 

def merge_to_get_xCenter_yCenter(DFs,df_processed):
    DFs_merged = {}
    df_processed = df_processed.drop(columns=['locationId','class']) # Should be unnecessary?
    for key,df in DFs.items():
        df['locationId'] = df['locationId'].astype(int)
        DFs_merged[key] = pd.merge(df,
                            df_processed,
                            on=[StateAttribute.FRAME.value,StateAttribute.OBJECT_ID.value],
                            how='left')
    return DFs_merged

def boxplot_horizon_VS_err_inner(dfs,names
        , locationId, clss, figsize=(8,6)):
    """
    Args:
        dfs: List of dataframes, each containing data for an algorithm.
        names: List with names of the algorithms.

    ..todo: notna() when getting the intersection of locations and classes is 
    unnecessary?
    """
    with sns.axes_style("darkgrid"): # https://stackoverflow.com/questions/29235206/changing-seaborn-style-in-subplots
                #ax = plt.figure(figsize=figsize).add_subplot()
                _, ax = plt.subplots(figsize=figsize)
    # same as other version but makes y-limits the same for plots at the same location
    dfs_tmp = []
    for ii,df in enumerate(dfs):
        df[f'Algorithm'] = names[ii]#f'algorithm {ii+1}'
        dfs_tmp.append(df[df['locationId']==locationId])# TODO: Move df['class']==clss below into here 
    df = pd.concat(dfs_tmp)

    df_tmp = df[df['class']==clss].filter(regex=(f'(err,h=*)|Algorithm'))
    dd = df_tmp.melt(id_vars=['Algorithm'])
    dd['variable'] = dd['variable'].str.extract(r'(\d+\.?\d+)')
    sns.boxplot(x='variable',y='value',data=dd,hue='Algorithm'
                ,whis=0*1000,medianprops=dict(color="white", alpha=1), ax = ax
                ,showfliers = False)

    # annotate axes
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator()) 
    ax.grid(b=True, which='minor', color='white', linewidth=0.5)
    ax.set_ylabel('prediction error [m]')
    ax.set_xlabel('prediction horizon $H$ [s]')
    plt.setp(ax.get_xticklabels(), rotation=90) 
    #plt.setp(ax.get_legend().get_texts(), fontsize=8) # for legend text
    #plt.setp(ax.get_legend().get_title(), fontsize=8) # for legend title
    #plt.setp(ax.get_legend(), handlelength=0.5)
    #ax.get_legend().set(handlelength=0.5)
    ax.legend(handlelength = 1,fontsize=8)
    #print(ax.get_legend().handlelength)

    return ax

def boxplot_horizon_VS_err(dfs,names,locationIds,classes,path,figsize=(8,6)):
    axs = {}
    for locId in locationIds:
        for clss in classes:
            axs[locId,clss] = boxplot_horizon_VS_err_inner(dfs,names
                                        , locId, clss, figsize)
    
    for locId in locationIds:
        for clss in classes:
            # make ylimits the same for both location 1 and location 2.
            if axs[1,clss].get_ylim()[1] < axs[2,clss].get_ylim()[1]:
                axs[1,clss].set_ylim(axs[2,clss].get_ylim())
            else:
                axs[2,clss].set_ylim(axs[1,clss].get_ylim())    

            # save figures
            if path is not None:
#                axs[ii,i].get_figure().savefig(path+f"boxplot_location={locId}_class={clss}.png"
#                                        ,bbox_inches="tight"
#                                        ,dpi=300
#                                       )
                axs[locId,clss].get_figure().savefig(path+f"boxplot_location={locId}_class={clss}.pgf",bbox_inches="tight")
            else:
                axs[locId,clss].set_title(clss)
    return axs


def plot_position_VS_FDE_perLocationAndClass_inner(df,name,locationId,clss, ax=None):
    # Assumes df_test contains columns 'err,h=1','err,h=2',...
   
    if ax is None:
        _, ax = plt.subplots()

    df = df[(df['locationId']==locationId) & (df['class']==clss)]
    df['algorithm'] = name
    
    PT.plot_background(locationId,ax)
    points = ax.scatter(df['xCenter'],df['yCenter']
                    ,s=0.05
                    ,c=df['err,h=4.8']
                    ,facecolor =df['err,h=4.8']
                    ,cmap='Reds')

    PT.set_plot_limits(locationId,ax)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax,points



def plot_position_VS_FDE_perLocationAndClass_comparison_new(df1,name1,df2,name2,locationIds,classes, path=None):
    
    # somehow use a dictionary to store the axes while still getting subplots
    axs = {}
    figs = {}
    for locId in locationIds:
        for clss in classes:
            fig,tmp = plt.subplots(1,2)
            figs[locId,clss] = fig
            points = []
            for i,(df,name) in enumerate(zip([df1,df2],[name1,name2])):
                axs[locId,clss,i] = tmp[i]
                _, ps = plot_position_VS_FDE_perLocationAndClass_inner(df.copy(),name,locId,clss, axs[locId,clss,i])
                points.append(ps)
            
            # adjust figure width
            w, h = fig.get_size_inches()
            zoom = PT.LatexWidth.IEEE_JOURNAL_TEXT.value/72.27/w
            fig.set_size_inches(w * zoom, h * zoom)
            fig.subplots_adjust(hspace=0, wspace=0)

            # adjust scatter color and color bars
            colorbar_max = max(df1.loc[df1['class']==clss, 'err,h=4.8'].max()
                                ,df2.loc[df2['class']==clss,'err,h=4.8'].max())

            ## scatter colors
            for ps in points:
                ps.set_clim(0,colorbar_max)

            ## colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=colorbar_max)
            fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
                        ,ax=[axs[locId,clss,0],axs[locId,clss,1]]
                        ,orientation='vertical'
                        ,shrink=0.3#0.325
                        ,label='meter') # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots AND https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
    return figs


def ADE_and_FDE_statistic_per_location(DFs,model_names,statistic,classes,path=None):
    for clss in classes:
        rows = []
        for name,df in zip(model_names,DFs):
            df = df[df['class']==clss]
            ADE_statistic = \
                (
                    df
                    .filter(regex=('err,h=*'))
                    .mean(axis=1)
                    .groupby(df['locationId'])
                    .aggregate(statistic)
                )
            FDE_statistic = \
                (
                    df[f'err,h=4.8']
                    .groupby(df['locationId'])
                    .aggregate(statistic)
                )
            
            string = name
            for locationId in [4,1,2,3]:
                string += f" & {np.round(ADE_statistic[locationId],2)} / {np.round(FDE_statistic[locationId],2)}"
            string += "\\\\"
            rows.append(string)
        print(f'ADE/FDE of {clss}:')
        print(rows)
    
#    if path is not None:
#        with open(path+"rows_for_table_with_state-of-the-art.txt",'w') as file:
#            for row in rows:
#                file.write(row+'\n')

def ADE_and_FDE_statistic_per_class_location_model(DFs,model_names,statistic,path=None,groups = ['class','locationId']):

    DFs = pd.concat(DFs,keys = model_names)
    DFs = DFs.rename_axis(index=['Model',None])
    columns = [f'err,h={x}' for x in np.round(np.arange(1,13)*0.4,1)]
    ADE_statistic = \
        (
            DFs
            .reset_index(level=0)
            .groupby(['Model']+groups)
            .apply(lambda g: g[columns].mean(axis=1).aggregate(statistic)) # calculate statistic of ADEs
            .unstack(level=0)
        )
    FDE_statistic = \
        (
            DFs
            .reset_index(level=0)
            .groupby(['Model']+groups)
            .apply(lambda g: g['err,h=4.8'].aggregate(statistic)) # statistic of FDEs
            .unstack(level=0)
        )
    
    table = np.round(ADE_statistic,2).astype(str) \
            + ' / '  \
            + np.round(FDE_statistic,2).astype(str)
    
    return table
    
def sanity_check_ADE_FDE_calculations(DFs,model_names):
    for df,m in zip(DFs.values(),model_names):
        print(f'Model = {m}')
        sanity_ADE_FDE(df)

def sanity_ADE_FDE(df):
    ADEs = {}
    FDEs = {}
    for locId in range(1,5):
        ADEs[locId] = []
        FDEs[locId] = []
        tmp = df[df['locationId']==locId]
        for objId in tmp['objectId'].unique():
            idx = tmp['objectId'] == objId
            ADEs[locId].append(tmp.loc[idx].filter(regex=('err,h=*')).mean(axis=1))
            FDEs[locId].append(tmp.loc[idx,'err,h=4.8'].iloc[-1])
        ADEs[locId] = np.round(np.mean(ADEs[locId]),2)
        FDEs[locId] = np.round(np.mean(FDEs[locId]),2)
    print(ADEs)
    print(FDEs)


def plot_position_VS_hit_miss_proportion(df,H,radius,locationId, clss,ax=None):
    df = df[(df['locationId'] == locationId) & (df['class'] == clss)]
    if ax is None:
        _,ax = plt.subplots()
    PT.plot_background(locationId,ax=ax)
    PT.set_plot_limits(locationId,ax)
    PT.despine(ax)
    hits = df[f'err,h={H}'] <= radius
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    N = complex(0,300)
    X,Y = np.mgrid[xlim[0]:xlim[1]:N,ylim[0]:ylim[1]:N]
    color_hits = np.zeros(X.shape)
    theta = 1
    pos_hits = df.loc[hits,['xCenter','yCenter']].to_numpy()
    for p in pos_hits:
        color_hits += np.sqrt((X-p[0])**2+(Y-p[1])**2) < theta
    color_miss =  np.zeros(X.shape)
    pos_miss = df.loc[~hits,['xCenter','yCenter']].to_numpy()
    for p in pos_miss:
        color_miss += np.sqrt((X-p[0])**2+(Y-p[1])**2)<theta
    denominator = color_hits+color_miss
    color_hits /= denominator
    color_miss /= denominator
    min_sqrd_dists = np.full(np.shape(X),np.inf)
    for p in np.row_stack([pos_hits,pos_miss]):
        D = (X-p[0])**2+(Y-p[1])**2
        min_sqrd_dists = np.min(np.stack([D,min_sqrd_dists]),axis=0)
    alpha = 7#theta
    A = np.exp(-alpha*min_sqrd_dists)
    I = np.array([color_miss,color_hits,np.zeros(X.shape),A]).T
    ax.imshow(np.fliplr(np.rot90(I,2)),extent=(*ax.get_xlim(),*ax.get_ylim()))
    return ax

def colormap_in_hit_miss_plot(ax=None):
    """Plots the colormap in polar coordinates."""
    if ax is None:
        _,ax = plt.subplots(figsize=(PT.LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27,1))
    N = 100
    theta = np.linspace(0,np.pi/2,N)
    r = np.linspace(0,1,N)
    R,T = np.meshgrid(r,theta)
    I = np.array([np.flipud(T),T,np.zeros(R.shape),np.exp(-7*R**2)]).T
    ax.imshow(np.rot90(I),extent=(0,1,0,1))
    ax.set_xlim((0,0.8))
    ax.set_xlabel('distance to closest datapoint [m]')
    ax.set_yticks([0,0.5,1])
    ax.set_yticklabels([r'0% / 100%',r'50% / 50%',r'100% / 0%'])
    ax.set_ylabel('success / fail')
    ax.set_aspect(0.5)
    return ax

def plot_position_VS_error_smooth_old(df,H,locationId, clss, quantile =0.95,ax=None):
    df = df[(df['locationId'] == locationId) & (df['class'] == clss)]
    if ax is None:
        fig,ax = plt.subplots()
    PT.plot_background(locationId,ax=ax)
    PT.set_plot_limits(locationId,ax)
    PT.despine(ax)
    adjust_width(fig,PT.LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27)

    model = sklearn.neighbors.RadiusNeighborsRegressor(radius=0.25)
    idx = df[f'err,h={H}'] < df[f'err,h={H}'].quantile(quantile)
    model = model.fit(df.loc[idx,['xCenter','yCenter']], df.loc[idx,f'err,h={H}'])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    N = complex(0,300)
    X,Y = np.mgrid[xlim[0]:xlim[1]:N,ylim[0]:ylim[1]:N]
    Z = model.predict(np.column_stack([X.flatten(),Y.flatten()])).reshape(X.shape)
    contours = ax.contourf(X,Y,Z,cmap='Reds')
    plt.colorbar(contours,ax=ax
                ,orientation='horizontal'
                ,shrink=1
                ,label='meter'
                ,pad=0.05)

    vmax = np.max(Z)
    return ax,vmax

def plot_position_VS_error_smooth(df,H,locationId, clss, quantile =0.95,ax=None,vmax=None):
    df = df[(df['locationId'] == locationId) & (df['class'] == clss)]
    if ax is None:
        _,ax = plt.subplots()
    PT.plot_background(locationId,ax=ax)
    PT.set_plot_limits(locationId,ax)
    PT.despine(ax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    N = complex(0,300)
    X,Y = np.mgrid[xlim[0]:xlim[1]:N,ylim[0]:ylim[1]:N]
    Z = position_VS_error_smooth(df ,np.column_stack([X.flatten(),Y.flatten()]),H)
    Z = Z.reshape(X.shape)
    contours = ax.contourf(X,Y,Z,cmap='Reds',vmin=0,vmax=vmax)
    vmax = np.nanmax(Z)
    return ax,vmax ,contours

def plot_position_VS_error_smooth_comparison(DFs,clss,H=4.8,locationId=2,n_contours = 7, fig_width=PT.LatexWidth.IEEE_JOURNAL_TEXT.value/72.27):
    for i,_ in enumerate(DFs):
        DFs[i] = DFs[i][(DFs[i]['locationId'] == locationId)&(DFs[i]['class'] == clss)]

    # Grid of positions to evaluate on
    xlim = [7,99] # for location 2, taken from PT.set_plot_limits
    ylim = [-49,0]
    N = complex(0,300)
    X,Y = np.mgrid[xlim[0]:xlim[1]:N,ylim[0]:ylim[1]:N]

    # Remove outlier errors
    idx = (DFs[0][f'err,h={H}'] < DFs[0][f'err,h={H}'].quantile(0.95)) \
         & (DFs[1][f'err,h={H}'] < DFs[1][f'err,h={H}'].quantile(0.95)) # the same data should be removed from both plots, otherwise it becomes more difficult to compare

    # Find maxmimum of each model for normalizing the plots
    Zs = [] 
    for i,df in enumerate(DFs):
        positions_train = df.loc[idx,['xCenter','yCenter']]
        errors_train = df.loc[idx,f'err,h={H}']
        positions_eval = np.column_stack([X.flatten(),Y.flatten()])
        Z = position_VS_error_smooth(positions_train, errors_train,positions_eval)
        Z = Z.reshape(X.shape)
        Zs.append(Z)
    vmax = max(map(np.nanmax,Zs))

    # plot
    fig,axs = plt.subplots(1,2)
    Cs = []
    for ax,Z in zip(axs,Zs):
        PT.plot_background(locationId,ax=ax)
        PT.set_plot_limits(locationId,ax)
        PT.despine(ax)
        contours = ax.contourf(X,Y,Z,cmap='Reds',vmin=0,vmax=vmax,levels=n_contours)
        Cs.append(contours)
    
    adjust_width(fig,fig_width)
    fig.colorbar(Cs[1]
                ,ax=axs
                ,orientation='vertical'
                ,shrink=0.25
                ,pad=0.025
                ,label='meter') # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots AND https://matplotlib.org/stable/tutorials/colors/colorbar_only.html
    return fig


def position_VS_error_smooth(X_train,y_train,X_eval):
    model = sklearn.neighbors.RadiusNeighborsRegressor(radius=0.25)
    model = model.fit(X_train, y_train)
    return model.predict(X_eval)
    

def adjust_width(fig,new_width):
    w, h = fig.get_size_inches()
    zoom = new_width/w
    fig.set_size_inches(w * zoom, h * zoom)
    fig.subplots_adjust(hspace=0, wspace=0)