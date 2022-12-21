
from dataclasses import dataclass
from typing import Any

class Dataset():
    name = 'InD'
    path = ... #'my-path/inD-dataset-v1.0/data'
    sampling_rate = 25

class DataProcessing(): # configuration common to all runs.
    sampling_rate = 2.5 # Hz
    max_past = 2.8 # seconds
    max_horizon = 4.8 # seconds

@dataclass
class Config():
    dataset = Dataset()
    paths: Any
    models: Any
    data_processing = DataProcessing()
    
# There is a major problem with this construction. Which is that is is unclear
# how the user must define the variables of Config.