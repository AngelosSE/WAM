import functools
import pickle
import hashlib
import json

def cache_on_disk(path,unhashed_key):
    """
    If results from decorated function exists on disk and use_storage is True,
    then replace the function evaluation with the results from disk.
    .. todo:
    - Is this sensitive to ordering in the dictionary?
    """
    def decorator(fun):
#        unhashed_key_as_yaml = omegaconf.OmegaConf.to_yaml(unhashed_key)\
#                                    .encode('utf-8')
        unhashed_key_as_json = json.dumps(unhashed_key)\
                                    .encode('utf-8')
        hashkey = hashlib.sha256(unhashed_key_as_json).hexdigest()
        filename = fun.__name__ + '_' + hashkey

        @functools.wraps(fun)
        def wrapper(*args,**kwargs):
            try:
                with open(path+filename,'rb') as file:
                    evaluation = pickle.load(file)
                print(f'From cache: {fun.__name__}')
            except FileNotFoundError:
                evaluation = fun(*args, **kwargs)
                with open(path+filename,'wb') as file:
                    pickle.dump(evaluation, file)
            return evaluation
        return wrapper
    return decorator

class cache_calls_on_disk():
    '''
    Stores on disk every call of a function. This is useful when you need to 
    work with a script where some functions are slow to evaluate. However,
    you have to be careful and clear the cache any time you change the order 
    of the decorated functions calls, otherwise you will get erroneaous results.
    '''
    def __init__(self,path):
        self.path = path
        self.call_count = 0

    def __call__(self, fun): # decorator
        @functools.wraps(fun)
        def wrapper(*args,**kwargs):
            self.call_count += 1
            filename = fun.__name__ + '_' + str(self.call_count)
            try:
                with open(self.path+filename,'rb') as file:
                    evaluation = pickle.load(file)
                print(f'From cache: {fun.__name__}')
            except FileNotFoundError:
                evaluation = fun(*args, **kwargs)
                with open(self.path+filename,'wb') as file:
                    pickle.dump(evaluation, file)
            return evaluation
        return wrapper

class temporary_decoration():
    def __init__(self,obj,method_name,decorator):
        self.method = getattr(obj,method_name)
        self.obj = obj
        self.method_name = method_name
        self.decorator = decorator
    
    def __enter__(self):
        setattr(self.obj,self.method_name, self.decorator(self.method))

    def __exit__(self,*_):
        setattr(self.obj,self.method_name,self.method)

class cache_call():
    def __init__(self):
        self.has_been_called = False
    def __call__(self,fun):
        def wrapper(*args,**kwargs):
            if self.has_been_called is True:
                return self.cache
            else:
                self.cache = fun(*args,**kwargs)
                self.has_been_called = True
                return self.cache
        return wrapper