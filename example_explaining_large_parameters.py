import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.base import BaseEstimator
import warnings
import plot_tools as PT

class Predictor(RadiusNeighborsRegressor):
    def __init__(self,radius=10000,theta=0.5):
        super().__init__(radius = radius, weights = self.similarity)
        self.theta = theta
        warnings.warn(f'Radius restricted to {radius}. This means that theta = 0, does not correspond to the average of the data.')
        
    def similarity(self,list_of_dists):
        similarities = []
        for dists in list_of_dists:
            s = np.exp(-self.theta*dists**2) # the large parameter I find for velocity in my paper could be due to this asserting that we do not get a value inbetween modes!
            s = s / np.sum(s)
            similarities.append(s)
        return similarities

def generate_trajectories(nominal_speed,nominal_position,sampling_time,models):
    times = np.arange(0,sampling_time*20,sampling_time)
    # generate trajectories, 5 with constant speed, 5 with linearly decaying speed
    np.random.seed(1)
    trajs = {}
    for name,model in models.items():
        trajs[name] = []
        for i in range(5): # generate trajectories with constant speed
            initial_speed = nominal_speed + np.random.normal()
            initial_position = nominal_position + np.random.normal()
            trajs[name].append(model(initial_position,initial_speed,times))
    return trajs

def create_examples(trajs,horizon,sampling_time):
    examples = []    
    for traj in trajs:
        n = int(horizon/sampling_time)
        current_pos = traj[:-n]
        future_pos = traj[n:]
        examples.append(np.vstack([current_pos,future_pos]).T) # column_stack?
    examples = np.vstack(examples)
    return examples

def main(path=None):
    # Backend
    nominal_speed = 20 # meter/second
    nominal_position = 0  # meter
    sampling_time = 0.4 # second
    rate_of_decay = 0.25 # 1/second
    models = {'constant': lambda p,s,t: p+s*t
              ,'decaying': lambda p,s,t: p +(s/rate_of_decay)*(1-np.exp(-rate_of_decay*t))}
    trajs = generate_trajectories(nominal_speed,nominal_position,sampling_time,models)
    horizon = 2 # second
    relation = {case:create_examples(trajs[case],horizon,sampling_time) for case in ['constant','decaying']}
    predictors ={'small' : Predictor(theta = 0.1),
                 'large' : Predictor(theta = 6)}
    tmp = np.vstack([*relation.values()])
    for predictor in predictors.values():
        predictor.fit(tmp[:,0].reshape(-1,1),tmp[:,1].reshape(-1,1))

    # Plot position versus future positions
    ## plot data
    fig,ax = plt.subplots()#(figsize=(LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27,3.5))#(figsize=fig_size(LatexWidth.IEEE_JOURNAL_COLUMN.value))
    handles = {}
    handles['data'] = {}
    for case,style in zip(['decaying','constant'],['+k','.k']):
        handles['data'][case], = ax.plot(*list(relation[case].T),style)

    ## Plot predictions
    handles['predictions'] = {}
    positions = np.linspace(0,max(tmp[:,0]),500)
    for (name,predictor),style in zip(predictors.items(),['-r','-b']):
        predictions = predictor.predict(positions.reshape(-1,1))
        handles['predictions'][name], = ax.plot(positions,predictions,style)

    ax.set_xlabel('Position [m]')
    ax.set_ylabel(f'Position {horizon} seconds later [m]')
    ax.legend(
        [
        handles['data']['decaying'],
        handles['data']['constant'],
        handles['predictions']['small'],
        handles['predictions']['large'],
        ],
        [
        'data, stopping vehicles',
        'data, continuing vehicles',
        'model, small parameter',
        'model, large parameter'
        ]
        )

    if path is not None:
        fig.savefig(path + 'example_explaining_large_parameters.png',dpi = 300)
        fig.tight_layout()
        fig.savefig(path + 'example_explaining_large_parameters.pgf')

    # Plot position versus vehicleID
    fig,ax = plt.subplots()
    
    handles = {
        'stopping' : ax.plot(trajs['constant'],range(1,5+1),'xk')[0],
        'continuing' : ax.plot(trajs['decaying'],range(6,10+1),'+k')[0]
        }
    
    ax.set_yticks(range(1,10+1))
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Vehicle ID')
    ax.legend([*handles.values()],['stopping vehicles','continuing vehicles'])
    if path is not None:
        fig.savefig(path + 'example_explaining_large_parameters_data.png',dpi = 300)
        #fig.tight_layout()
        #fig.savefig(path + '/example_explaining_large_parameters_data.pgf')
        
def plot_position_VS_future_position(relation,predictors,horizon,tmp,path = None):
    ## plot data
    fig = plt.figure(figsize=(PT.LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27*0.95,1.5))
    ax = fig.add_axes((0.15,0.21,0.85,0.79))
    handles = {}
    handles['data'] = {}
    for case,style in zip(['decaying','constant'],['+k','.k']):
        handles['data'][case], = ax.plot(*list(relation[case].T),style,markersize=4)

    ## Plot predictions
    handles['predictions'] = {}
    positions = np.linspace(0,max(tmp[:,0]),500)
    for (name,predictor),style in zip(predictors.items(),['-b','-r']):
        predictions = predictor.predict(positions.reshape(-1,1))
        handles['predictions'][name], = ax.plot(positions,predictions,style)

    ax.set_xlabel('Position [m]')
    ax.set_ylabel(f'Position {horizon} s later [m]')
    ax.legend(
        [
        handles['data']['decaying'],
        handles['data']['constant'],
        handles['predictions']['small'],
        handles['predictions']['large'],
        ],
        [
        'stopping vehicles',
        'continuing vehicles',
        'small parameter',
        'large parameter'
        ]
        #,loc = 'lower right'
        ,handlelength=0.5
        ,borderaxespad=0,handletextpad=0.5,borderpad=0.1,labelspacing=0.3
        )
    if path is not None:
        fig.savefig(path + 'example_explaining_large_parameters.png',dpi = 300,bbox_inches="tight")
        #fig.tight_layout()
        #fig.savefig(path + 'example_explaining_large_parameters.pgf',bbox_inches="tight")

def plot_position_VS_id(trajs,path=None):
    fig = plt.figure(figsize=(PT.LatexWidth.IEEE_JOURNAL_COLUMN.value/72.27*0.95,1.5))
    ax = fig.add_axes((0.15,0.21,0.85,0.79))
    
    handles = {
        'stopping' : ax.plot(trajs['constant'],range(1,5+1),'+k',markersize=4)[0],
        'continuing' : ax.plot(trajs['decaying'],range(6,10+1),'.k',markersize=4)[0]
        }
    
    ax.set_yticks(range(1,10+1))
    ax.set_xlabel('Position [m]')
    ax.set_ylabel('Vehicle ID')
    ax.legend([*handles.values()],['stopping vehicles','continuing vehicles']
                ,borderaxespad=0.3,handletextpad=0.25,borderpad=0.1,labelspacing=0.3)
    if path is not None:
        fig.savefig(path + 'example_explaining_large_parameters_data.png',dpi = 300,bbox_inches="tight")
        #fig.tight_layout()
        #fig.savefig(path + '/example_explaining_large_parameters_data.pgf')

def main_result(path=None):
    # Backend
    nominal_speed = 20 # meter/second
    nominal_position = 0  # meter
    sampling_time = 0.4 # second
    rate_of_decay = 0.25 # 1/second
    models = {'constant': lambda p,s,t: p+s*t
              ,'decaying': lambda p,s,t: p +(s/rate_of_decay)*(1-np.exp(-rate_of_decay*t))}
    trajs = generate_trajectories(nominal_speed,nominal_position,sampling_time,models)
    horizon = 2 # second
    relation = {case:create_examples(trajs[case],horizon,sampling_time) for case in ['constant','decaying']}
    predictors ={
                'large' : Predictor(theta = 6)
                ,'small' : Predictor(theta = 0.5)
                 }
    tmp = np.vstack([*relation.values()])
    for predictor in predictors.values():
        predictor.fit(tmp[:,0].reshape(-1,1),tmp[:,1].reshape(-1,1))

    # Plotting
    plt.rcParams['text.usetex'] = True
    plot_position_VS_future_position(relation,predictors,horizon,tmp,path)
    plot_position_VS_id(trajs,path)
    plt.rcParams['text.usetex'] = False


if __name__ == '__main__':
    plt.close('all')
    #main(save=1)
    main_result()
    plt.show()










