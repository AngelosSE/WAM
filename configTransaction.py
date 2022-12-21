import configMy
import configScout
from config import Config
import os 

class Paths():
    save_tables = ... # path as string to save tabular results to disk, else None to ignore
    save_figures =  ... # path as string to save figure results to disk, else None to ignore
    cache = ... # path as string to save cached results to disk, else None to ignore

if (Paths.cache is None) or not (os.path.isfile(Paths.cache)): # This code should ideally be placed in transaction_results.main. However, it is impossible since the decoration must happen before configMy.models() is called. 
    import models
    models.WAM.predict = models.time_prediction()(models.WAM.predict)

cfgMy = Config(type('PathsMy',(Paths,),{})
                ,configMy.models())
cfgMy.name = 'my' # required in pipeline_components.run_data_processing

cfgScout = Config(type('PathsScout',(Paths,),{})
                ,configScout.models())
cfgScout.name = 'scout'# required in pipeline_components.run_data_processing