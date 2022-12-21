from models import Predictor_constVel
import numpy as np
from definitions import Location, RoadUserClass, StateAttribute
import models_transaction as Models


def models():
    horizons =  [1,2,3,4,5,6,7,8,9,10,11,12] # this is a restriction common to all models
    models = [] 
    models.append(
        Predictor_constVel(
            horizons = horizons
            ,restrictions={(StateAttribute.LOCATION_ID,0): np.array([Location.ONE.value,Location.TWO.value])
                            ,(StateAttribute.CLASS,0): np.array([RoadUserClass.VEHICLE.value,RoadUserClass.BICYCLE.value,RoadUserClass.PEDESTRIAN.value])}
            ,name='constant_velocity')
        )
    
    radius = 15 # common to all WAM models
    max_iter = 400 # common to all MLP models
 
    models.extend(models_at_location1(horizons,radius,max_iter))
    models.extend(models_at_location2(horizons,radius,max_iter))

    return models

def models_at_location1(horizons,radius,max_iter):
    models = []
    restrictions = {(StateAttribute.LOCATION_ID,0):np.array([Location.ONE.value])}
    for roadUser, parameters in [(RoadUserClass.VEHICLE,[0.5,0.5,1.0,50.0])
                                ,(RoadUserClass.BICYCLE,[0.25,0.25,20.0,50.0])
                                 ,(RoadUserClass.PEDESTRIAN,[0.25,0.25,20.0,50])]:
        models.append(Models.WAM(
                horizons
                ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,radius = radius
                ,name='WAM'
                ,parameters=parameters
            ))

    for roadUser, parameter in [(RoadUserClass.VEHICLE,500)
                                ,(RoadUserClass.BICYCLE,225)
                                ,(RoadUserClass.PEDESTRIAN,500)]:
        models.append(Models.MLP(
            horizons = horizons
            ,restrictions = restrictions | {(StateAttribute.CLASS,0):np.array([roadUser.value])}
            ,hidden_layer_sizes=parameter
            ,max_iter=max_iter
            ,name = 'MLP'
            ))
    return models

def models_at_location2(horizons,radius,max_iter):
    models = []
    restrictions = {(StateAttribute.LOCATION_ID,0):np.array([Location.TWO.value])}
    for roadUser,parameters in [(RoadUserClass.VEHICLE,[0.5,0.5,1.0,200.0])
                                ,(RoadUserClass.BICYCLE,[0.25,0.25,1.0,100.0])
                                ,(RoadUserClass.PEDESTRIAN,[0.1,0.1,50.0,50.0])]:
        models.append(Models.WAM(
                horizons
                ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,radius = radius
                ,name='WAM'
                ,parameters=parameters
            ))

    for roadUser,parameter in [(RoadUserClass.VEHICLE,400)
                                ,(RoadUserClass.BICYCLE,350)
                                ,(RoadUserClass.PEDESTRIAN,150)]:
        models.append(Models.MLP(
                horizons = horizons
                ,restrictions = restrictions | {(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,hidden_layer_sizes=parameter
                ,max_iter=max_iter
                ,name = 'MLP'
            ))
    return models