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
            ,restrictions={(StateAttribute.LOCATION_ID,0): np.array([Location.ONE.value,Location.TWO.value,Location.THREE.value,Location.FOUR.value])
                            ,(StateAttribute.CLASS,0): np.array([RoadUserClass.VEHICLE.value,RoadUserClass.BICYCLE.value,RoadUserClass.PEDESTRIAN.value])}
            ,name='constant_velocity')
        )
    
    radius = 100 # common to all WAM models
    max_iter = 4000 # common to all MLP models
        
    models.extend(models_at_location1(horizons,radius,max_iter))
    models.extend(models_at_location2(horizons,radius,max_iter))
    models.extend(models_at_location3(horizons,radius,max_iter))
    models.extend(models_at_location4(horizons,radius,max_iter))
    return models

def models_at_location1(horizons,radius,max_iter):
    models = []
    restrictions = {(StateAttribute.LOCATION_ID,0):np.array([Location.ONE.value])}
    for roadUser, parameters in [(RoadUserClass.VEHICLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.BICYCLE,[0.5,0.5,20.0,50.0])
                                ,(RoadUserClass.PEDESTRIAN,[0.25,0.25,50.0,75.0])]:
        models.append(Models.WAM(
            horizons
            ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
            ,radius = radius
            ,name='WAM'
            ,parameters=parameters
            ))

    for roadUser, parameter in [(RoadUserClass.VEHICLE,350)
                                ,(RoadUserClass.BICYCLE,500)
                                ,(RoadUserClass.PEDESTRIAN,300)]:
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
    for roadUser,parameters in [(RoadUserClass.VEHICLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.BICYCLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.PEDESTRIAN,[0.25,0.25,50.0,75.0])]:
        models.append(Models.WAM(
                horizons
                ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,radius = radius
                ,name='WAM'
                ,parameters=parameters
            ))

    for roadUser,parameter in [(RoadUserClass.VEHICLE,300)
                                ,(RoadUserClass.BICYCLE,250)
                                ,(RoadUserClass.PEDESTRIAN,450)]:
        models.append(Models.MLP(
                horizons = horizons
                ,restrictions = restrictions | {(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,hidden_layer_sizes=parameter
                ,max_iter=max_iter
                ,name = 'MLP'
            ))
    return models

def models_at_location3(horizons,radius,max_iter):
    models = []
    restrictions = {(StateAttribute.LOCATION_ID,0):np.array([Location.THREE.value])}
    for roadUser,parameters in [(RoadUserClass.VEHICLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.BICYCLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.PEDESTRIAN,[0.25,0.25,50.0,75.0])]:
        models.append(Models.WAM(
                horizons
                ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,radius = radius
                ,name='WAM'
                ,parameters=parameters
            ))

    for roadUser,parameter in [(RoadUserClass.VEHICLE,350)
                                ,(RoadUserClass.BICYCLE,350)
                                ,(RoadUserClass.PEDESTRIAN,450)]:
        models.append(Models.MLP(
                horizons = horizons
                ,restrictions = restrictions | {(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,hidden_layer_sizes=parameter
                ,max_iter=max_iter
                ,name = 'MLP'
            ))
    return models

def models_at_location4(horizons,radius,max_iter):
    models = []
    restrictions = {(StateAttribute.LOCATION_ID,0):np.array([Location.FOUR.value])}
    for roadUser,parameters in [(RoadUserClass.VEHICLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.BICYCLE,[0.25,0.25,1.0,50.0])
                                ,(RoadUserClass.PEDESTRIAN,[0.25,0.25,50.0,75.0])]:
        models.append(Models.WAM(
                horizons
                ,restrictions = restrictions |{(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,radius = radius
                ,name='WAM'
                ,parameters=parameters
            ))

    for roadUser,parameter in [(RoadUserClass.VEHICLE,300)
                                ,(RoadUserClass.BICYCLE,200)
                                ,(RoadUserClass.PEDESTRIAN,400)]:
        models.append(Models.MLP(
                horizons = horizons
                , restrictions = restrictions | {(StateAttribute.CLASS,0):np.array([roadUser.value])}
                ,hidden_layer_sizes=parameter
                ,max_iter=max_iter
                ,name = 'MLP'
            ))
    return models    