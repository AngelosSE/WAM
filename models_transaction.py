from definitions import StateAttribute
import models
import functools
import numpy as np

ATTRIBUTES_IN_DOMAIN = ((StateAttribute.X_CENTER,0), 
                        (StateAttribute.Y_CENTER,0),
                        (StateAttribute.SPEED,0),
                        (StateAttribute.X_ORIENTATION,0), 
                        (StateAttribute.Y_ORIENTATION,0))

STATEATTRS_IN_CODOMAIN = [(StateAttribute.X_CENTER),
                          (StateAttribute.Y_CENTER)]
STATEATTRS_IN_CODOMAIN_DISPL = [StateAttribute.X_DISPLACEMENT,
                                StateAttribute.Y_DISPLACEMENT]

class WAM(models.WAM_over_horizon_position):
    def __init__(self
            ,horizons
            ,restrictions
            ,radius
            ,name
            ,parameters):
        diff = functools.partial(models.diff_mix
            ,attrs_standard_diff=ATTRIBUTES_IN_DOMAIN[:3])
        super().__init__( 
            attributes_in_domain = ATTRIBUTES_IN_DOMAIN
            , horizons = horizons
            ,restrictions = restrictions
            ,weights=functools.partial(models.weights,
                        diff = diff,
                        transformation = models.transf_identity)
            ,weights_parameters={'parameters':np.array(parameters)}
            ,radius = radius
            ,name=name
            ,stateAttrs_in_codomain=STATEATTRS_IN_CODOMAIN
            ,stateAttrs_codomain_displ=STATEATTRS_IN_CODOMAIN_DISPL
            ,attributes_in_balltree=((StateAttribute.X_CENTER,0),
                                    (StateAttribute.Y_CENTER,0)))

class MLP(models.MyMLP):
    def __init__(self,horizons,restrictions,name,hidden_layer_sizes,max_iter=400):
        super().__init__(
            attributes_in_domain=ATTRIBUTES_IN_DOMAIN
            ,horizons=horizons
            , restrictions = restrictions
            ,hidden_layer_sizes = hidden_layer_sizes
            ,max_iter=max_iter
            ,name=name
            ,stateAttrs_in_codomain=STATEATTRS_IN_CODOMAIN
            ,stateAttrs_in_codomain_displ=STATEATTRS_IN_CODOMAIN_DISPL)