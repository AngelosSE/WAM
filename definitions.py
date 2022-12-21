from enum import Enum
from numpy.typing import NDArray

class StateAttribute(str,Enum):
     X_CENTER = 'xCenter'
     Y_CENTER = 'yCenter'
     X_VELOCITY = 'xVelocity'
     Y_VELOCITY = 'yVelocity'
     HEADING = 'heading'
     SPEED = 'speed'
     X_ORIENTATION = 'xOrientation'
     Y_ORIENTATION = 'yOrientation'
     FRAME = 'frame'
     LOCATION_ID = 'locationId'
     CLASS = 'class'
     OBJECT_ID = 'objectId'
     X_DISPLACEMENT = 'xDisplacement'
     Y_DISPLACEMENT = 'yDisplacement'
     X_CENTER_EGO = 'xCenterEgo'
     Y_CENTER_EGO = 'yCenterEgo'
     LAMBDA = 'lambda'
     LAMBDA_DISPLACEMENT = 'lambdaDisplacement'
     LANE = 'lane'
     ACC_CURV = 'accCurv'
     ACCELERATION = 'acceleration'
     ORIGINAL_OBJECT_ID = 'originalObjectId'

class RoadUserClass(str,Enum):
    VEHICLE = 'vehicle'
    BICYCLE = 'bicycle'
    PEDESTRIAN = 'pedestrian'

class Location(int,Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4

Attribute = tuple[StateAttribute,int]
Data =  dict[
            Attribute, # int is time index relative current.
            NDArray#ArrayLike # shape = (n_data,)
            ]