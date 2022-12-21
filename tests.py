import functools
import unittest
import numpy as np
import pandas as pd
import sklearn
from data_processing import create_data, discard_short_trajs
import data_processing as DP
from data_processing_utils import restrict
from definitions import StateAttribute
import models as my_models
import tempfile

class Test_DataProcessing_merge_into_DF(unittest.TestCase):
    def test1(self):
        recordingMeta = []
        tracks = []
        tracksMeta = []
        recordingMeta.append(pd.DataFrame([[0,1,25,1]]
            ,columns=['recordingId','locationId','frameRate','orthoPxToMeter']))
        tracks.append(pd.DataFrame(np.column_stack([6*[0],3*[0]+3*[1],2*[0,1,2],np.double(np.random.randint(10,size=(6,5))),range(6),range(6)]),
            columns=['recordingId','trackId','frame','xCenter','yCenter','xVelocity','yVelocity','heading','xAcceleration','yAcceleration']))
        tracksMeta.append(pd.DataFrame([[0,'pedestrian'],[1,'vehicle']],
            columns=['trackId','class']))

        recordingMeta.append(pd.DataFrame([[1,2,25,1]]
            ,columns=['recordingId','locationId','frameRate','orthoPxToMeter']))
        tracks.append(pd.DataFrame(np.column_stack([6*[1],3*[0]+3*[1],2*[0,1,2],np.random.randint(10,size=(6,5)),range(6),range(6)]),
            columns=['recordingId','trackId','frame','xCenter','yCenter','xVelocity','yVelocity','heading','xAcceleration','yAcceleration']))
        tracksMeta.append(pd.DataFrame([[0,'bicycle'],[1,'vehicle']],
            columns=['trackId','class']))

        tmp =['xCenter','yCenter','xVelocity','yVelocity','heading','xAcceleration','yAcceleration']
        tmp = pd.concat([tracks[0][tmp],tracks[1][tmp]],axis=0)
        expected = pd.DataFrame({'recordingId':6*[0]+6*[1]
                                ,'locationId':6*[1]+6*[2]
                                ,'frameRate':12*[25.0]
                                ,'originalObjectId':2*(3*[0]+3*[1])
                                ,'objectId':3*[0]+3*[1]+3*[2]+3*[3]
                                ,'class':3*['pedestrian']+3*['vehicle']+3*['bicycle']+3*['vehicle']
                                ,'orthoPxToMeter':12*[1.0]
                                ,'frame':4*[0,1,2]
                                ,'xAcceleration':2*list(range(6))
                                ,'yAcceleration':2*list(range(6))}
                                | tmp.to_dict('list'))
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i in [0,1]:
                recordingMeta[i].to_csv(tmpdirname+f'/0{i}_recordingMeta.csv',index=False)
                tracks[i].to_csv(tmpdirname+f'/0{i}_tracks.csv',index=False)
                tracksMeta[i].to_csv(tmpdirname+f'/0{i}_tracksMeta.csv',index=False)
            result = DP.merge_into_DF(tmpdirname,recordingIds=[0,1])


        tmp= [c for c in result.columns if c != 'class']
        self.assertTrue(np.allclose(result[tmp],expected[tmp]))
        self.assertTrue(np.all(result['class']==expected['class']))
        
class Test_DataProcessing_discard_short_trajs(unittest.TestCase):
    def test1(self):
        sampling_rate = 25.0
        max_past = 2.8
        max_horizon = 4.8
        M = int(sampling_rate*(max_past+max_horizon))
        N =(M-1)
        df = pd.DataFrame({'objectId':(M+1)*[1] # +1 for "current time"
                                    + (N+1)*[2]
                           ,'dummy':(M+1+N+1)*[0]})
        result = discard_short_trajs(df,max_past,max_horizon,sampling_rate)
        expected = df[:M+1]
        self.assertTrue(result.equals(expected))

class Test_DataProcessing_permuations(unittest.TestCase):
    def test1(self):
        expected = []
        for i in [0,1]:
            for ii in [0,1]:
                for iii in [0,1]:
                    for iiii in [0,1]:
                        expected.append([i,ii,iii,iiii])
        result = DP.permutations(4)
        self.assertEqual(set(map(tuple,expected)),set(map(tuple,result)))

class Test_DataProcessing_add_orientation(unittest.TestCase):
    def test1(self):
        ''''Identity test'''
        theta = np.linspace(0,2*np.pi)
        df = pd.DataFrame(np.column_stack([np.cos(theta),np.sin(theta)])
                            ,columns=['xVelocity','yVelocity'])
        expected = df.copy().to_numpy()
        result = DP.add_orientation(df)[['xOrientation','yOrientation']].to_numpy()
        self.assertTrue(np.allclose(expected,result))

    def test2(self):
        '''Normalization'''
        theta = np.linspace(0,2*np.pi)
        expected = np.column_stack([np.cos(theta),np.sin(theta)])
        lengths = 1+np.random.uniform(size=(theta.size,1))
        df = pd.DataFrame(lengths*expected
                        ,columns=['xVelocity','yVelocity'])
        result = DP.add_orientation(df)[['xOrientation','yOrientation']].to_numpy()
        self.assertTrue(np.allclose(expected,result))
    
    def test3(self):
        '''Test rounding'''
        theta = np.array([0,np.pi/4,np.pi/2,np.pi*5/8,np.pi*6/8])
        lengths = np.array([1,0.005,1,0.005,0.005]).reshape(-1,1)
        df = pd.DataFrame(lengths*np.column_stack([np.cos(theta),np.sin(theta)])
                        ,columns=['xVelocity','yVelocity'])
        result = DP.add_orientation(df)[['xOrientation','yOrientation']].to_numpy()
        theta_expected = theta[[0,0,2,2,2]]
        expected = np.column_stack([np.cos(theta_expected),np.sin(theta_expected)])
        self.assertTrue(np.allclose(expected,result))
    

class Test_DataProcessing_create_data(unittest.TestCase):

    def test1(self):
        n_past = 1
        n_future = 0
        df = np.column_stack([
            3*[1]+3*[2],
            [10,11,12,30,31,32],
            [20,21,22,40,41,42]
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,1,2,2]),
                    (StateAttribute.X_CENTER,-1) : np.array([10,11,30,31]),
                    (StateAttribute.Y_CENTER,-1) : np.array([20,21,40,41]),
                    (StateAttribute.X_CENTER,0) : np.array([11,12,31,32]),
                    (StateAttribute.Y_CENTER,0) : np.array([21,22,41,42])}
        const_state_attrs = [StateAttribute.OBJECT_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))
        
    def test2(self):
        n_past = 0
        n_future = 1
        df = np.column_stack([
            3*[1]+3*[2],
            [10,11,12,30,31,32],
            [20,21,22,40,41,42]
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,1,2,2]),
                    (StateAttribute.X_CENTER,0) : np.array([10,11,30,31]),
                    (StateAttribute.Y_CENTER,0) : np.array([20,21,40,41]),
                    (StateAttribute.X_CENTER,1) : np.array([11,12,31,32]),
                    (StateAttribute.Y_CENTER,1) : np.array([21,22,41,42]),
                    (StateAttribute.X_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.Y_DISPLACEMENT,1):np.array(4*[1])
                    }

        const_state_attrs = [StateAttribute.OBJECT_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))

    def test3(self):
        n_past = 1
        n_future = 1
        df = np.column_stack([
            3*[1]+3*[2],
            [10,11,12,30,31,32],
            [20,21,22,40,41,42]
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,2]),
                    (StateAttribute.X_CENTER,-1) : np.array([10,30]),
                    (StateAttribute.Y_CENTER,-1) : np.array([20,40]),
                    (StateAttribute.X_CENTER,0) : np.array([11,31]),
                    (StateAttribute.Y_CENTER,0) : np.array([21,41]),
                    (StateAttribute.X_CENTER,1) : np.array([12,32]),
                    (StateAttribute.Y_CENTER,1) : np.array([22,42]),
                    (StateAttribute.X_DISPLACEMENT,1):np.array(2*[1]),
                    (StateAttribute.Y_DISPLACEMENT,1):np.array(2*[1])
                    }
        const_state_attrs = [StateAttribute.OBJECT_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))


    def test4(self):
        n_past = 2
        n_future = 2
        df = np.column_stack([
            6*[1]+6*[2],
            [10,11,12,13,14,15,30,31,32,33,34,35],
            [20,21,22,23,24,25,40,41,42,43,44,45]
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,1,2,2]),
                    (StateAttribute.X_CENTER,-2) : np.array([10,11,30,31]),
                    (StateAttribute.Y_CENTER,-2) : np.array([20,21,40,41]),
                    (StateAttribute.X_CENTER,-1) : np.array([11,12,31,32]),
                    (StateAttribute.Y_CENTER,-1) : np.array([21,22,41,42]),
                    (StateAttribute.X_CENTER,0) : np.array([12,13,32,33]),
                    (StateAttribute.Y_CENTER,0) : np.array([22,23,42,43]),
                    (StateAttribute.X_CENTER,1) : np.array([13,14,33,34]),
                    (StateAttribute.Y_CENTER,1) : np.array([23,24,43,44]),
                    (StateAttribute.X_CENTER,2) : np.array([14,15,34,35]),
                    (StateAttribute.Y_CENTER,2) : np.array([24,25,44,45]),
                    (StateAttribute.X_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.Y_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.X_DISPLACEMENT,2):np.array(4*[2]),
                    (StateAttribute.Y_DISPLACEMENT,2):np.array(4*[2])}
        const_state_attrs = [StateAttribute.OBJECT_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))

    def test5(self):
        n_past = 2
        n_future = 2
        df = np.column_stack([
            6*[1]+6*[2],
            [10,11,12,13,14,15,30,31,32,33,34,35],
            [20,21,22,23,24,25,40,41,42,43,44,45],
            [100,110,120,130,140,150,300,310,320,330,340,350]
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER,
                    StateAttribute.FRAME]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,1,2,2]),
                    (StateAttribute.X_CENTER,-2) : np.array([10,11,30,31]),
                    (StateAttribute.Y_CENTER,-2) : np.array([20,21,40,41]),
                    (StateAttribute.FRAME,-2) : np.array([100,110,300,310]),
                    (StateAttribute.X_CENTER,-1) : np.array([11,12,31,32]),
                    (StateAttribute.Y_CENTER,-1) : np.array([21,22,41,42]),
                    (StateAttribute.FRAME,-1) : np.array([110,120,310,320]),
                    (StateAttribute.X_CENTER,0) : np.array([12,13,32,33]),
                    (StateAttribute.Y_CENTER,0) : np.array([22,23,42,43]),
                    (StateAttribute.FRAME,-0) : np.array([120,130,320,330]),
                    (StateAttribute.X_CENTER,1) : np.array([13,14,33,34]),
                    (StateAttribute.Y_CENTER,1) : np.array([23,24,43,44]),
                    (StateAttribute.FRAME,1) : np.array([130,140,330,340]),
                    (StateAttribute.X_CENTER,2) : np.array([14,15,34,35]),
                    (StateAttribute.Y_CENTER,2) : np.array([24,25,44,45]),
                    (StateAttribute.FRAME,2) : np.array([140,150,340,350]),
                    (StateAttribute.X_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.Y_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.X_DISPLACEMENT,2):np.array(4*[2]),
                    (StateAttribute.Y_DISPLACEMENT,2):np.array(4*[2])}
        const_state_attrs = [StateAttribute.OBJECT_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))

    def test6(self):
        n_past = 2
        n_future = 2
        df = np.column_stack([
            6*[1]+6*[2],
            6*[3]+6*[4],
            [10,11,12,13,14,15,30,31,32,33,34,35],
            [20,21,22,23,24,25,40,41,42,43,44,45],
            ])
        state_attrs = [StateAttribute.OBJECT_ID,
                    StateAttribute.LOCATION_ID,
                    StateAttribute.X_CENTER,
                    StateAttribute.Y_CENTER]
        df = pd.DataFrame(df,
            columns = [stateAttr.value for stateAttr in state_attrs])
        expected = {(StateAttribute.OBJECT_ID,0) : np.array([1,1,2,2]),
                    (StateAttribute.LOCATION_ID,0) : np.array([3,3,4,4]),
                    (StateAttribute.X_CENTER,-2) : np.array([10,11,30,31]),
                    (StateAttribute.Y_CENTER,-2) : np.array([20,21,40,41]),
                    (StateAttribute.X_CENTER,-1) : np.array([11,12,31,32]),
                    (StateAttribute.Y_CENTER,-1) : np.array([21,22,41,42]),
                    (StateAttribute.X_CENTER,0) : np.array([12,13,32,33]),
                    (StateAttribute.Y_CENTER,0) : np.array([22,23,42,43]),
                    (StateAttribute.X_CENTER,1) : np.array([13,14,33,34]),
                    (StateAttribute.Y_CENTER,1) : np.array([23,24,43,44]),
                    (StateAttribute.X_CENTER,2) : np.array([14,15,34,35]),
                    (StateAttribute.Y_CENTER,2) : np.array([24,25,44,45]),
                    (StateAttribute.X_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.Y_DISPLACEMENT,1):np.array(4*[1]),
                    (StateAttribute.X_DISPLACEMENT,2):np.array(4*[2]),
                    (StateAttribute.Y_DISPLACEMENT,2):np.array(4*[2])}
        const_state_attrs = [StateAttribute.OBJECT_ID,
                             StateAttribute.LOCATION_ID]
        result = create_data(df,state_attrs,const_state_attrs,n_past,n_future)
        #print(f'expected =\n{np.column_stack([expected[k] for k in set(expected)])}')
        #print(f'result=\n{np.column_stack([result[k] for k in set(result)])}')
        self.assertTrue(all([np.all(expected[k]==result[k].flatten()) 
                                for k in result.keys()]))

class Test_project_restrict_select(unittest.TestCase):
    def setUp(self):
        state_attributes  = list(StateAttribute)
        times = range(-2,2+1)

        data = {}
        for t in times:
            for stateAttr in state_attributes:
                data[stateAttr,t] = np.array([1,2,3]) #np.random.default_rng().random(size=(3,1))
        self.data = data
        
    def test_restrict_1(self):
        restrictions = {(StateAttribute.OBJECT_ID,0) : np.array([1]),
                        (StateAttribute.LOCATION_ID,0) : np.array([3])}
        
        result = restrict(self.data,restrictions)
        self.assertTrue(sum(map(np.size, result.values()))==0)

    def test_restrict_2(self):
        data = self.data.copy()
        data[(StateAttribute.LOCATION_ID,0)][0] = 3
        restrictions = {(StateAttribute.OBJECT_ID,0) : np.array([1]),
                        (StateAttribute.LOCATION_ID,0) : np.array([3])}
        expected = {k:v[:1] for k,v in data.items()}
        result = restrict(self.data,restrictions)
        self.assertTrue(expected==result)

    def test_restrict_3(self):
        data = self.data.copy()
        restrictions = {(StateAttribute.OBJECT_ID,0) : np.array([1,2])}
        expected = {k:v[:2] for k,v in data.items()}
        result = restrict(self.data,restrictions)
        self.assertTrue(all([np.all(v1==v2) for v1,v2 
                            in zip(expected.values(),result.values())]))

class Test_weights(unittest.TestCase):

    def test_diff(self):
        Xbar = {(StateAttribute.X_CENTER,0) : np.array([1,2,3,4]),
                (StateAttribute.Y_CENTER,0) : np.array([5,6,7,8])}
        x = {(StateAttribute.X_CENTER,0) : np.array([0]),
            (StateAttribute.Y_CENTER,0) : np.array([0])}
        expected = np.column_stack([[1,2,3,4],[5,6,7,8]])
        result = my_models.diff(Xbar,x)
        self.assertTrue(np.allclose(expected,result))

    def test_diff_curr_orientation(self):
        stateAttrs =list(StateAttribute)
        attrs = list(zip(stateAttrs,len(stateAttrs)*[0]))
        Xbar = dict.fromkeys(attrs)
        Xbar[StateAttribute.X_ORIENTATION,0] = np.array([1,0,1/np.sqrt(2),-1])
        Xbar[StateAttribute.Y_ORIENTATION,0] = np.array([0,1,1/np.sqrt(2),0])
        x = {(StateAttribute.X_ORIENTATION,0) : np.array([1]),
            (StateAttribute.Y_ORIENTATION,0) : np.array([0])}
        expected = np.array([0,np.pi/2,np.pi/4,np.pi])
        result = my_models.diff_curr_orientation(Xbar,x)
        self.assertTrue(np.allclose(expected,result))
        
    def test_diff_mix(self): 
        stateAttrs =list(StateAttribute)
        attrs = list(zip(stateAttrs,len(stateAttrs)*[0]))
        Xbar = dict.fromkeys(attrs)
        Xbar[StateAttribute.X_ORIENTATION,0] = np.array([1,0,1/np.sqrt(2),-1])
        Xbar[StateAttribute.Y_ORIENTATION,0] = np.array([0,1,1/np.sqrt(2),0])
        Xbar[StateAttribute.X_CENTER,0] = np.array([1,2,3,4])
        x = {(StateAttribute.X_ORIENTATION,0) : np.array([1]),
            (StateAttribute.Y_ORIENTATION,0) : np.array([0]),
            (StateAttribute.X_CENTER,0) : np.array([0])}
        expected = np.column_stack([[1,2,3,4],[0,np.pi/2,np.pi/4,np.pi]])
        result = my_models.diff_mix(Xbar,x,[(StateAttribute.X_CENTER,0)])
        self.assertTrue(np.allclose(expected,result))

    def test_normalized_gaussian(self): 
        x = np.random.default_rng().random(size=(10,4))
        parameters = np.random.default_rng().random(size=(4,))
        y = np.exp(-np.sum(parameters*x**2,axis=1))
        expected = y/np.sum(y)
        result1 = my_models.normalized_gaussian(x,parameters,
            use_numerical_precaution=1)
        result2 = my_models.normalized_gaussian(x,parameters,
            use_numerical_precaution=0)
        self.assertTrue(np.allclose(expected,result1) and np.allclose(expected,result2))

    def test_transf_pos_to_displ(self):
        stateAttrs =list(StateAttribute)
        attrs = list(zip(stateAttrs,len(stateAttrs)*[0]))
        data = dict.fromkeys(attrs)
        data[StateAttribute.X_CENTER,0] = np.array([10,8])
        data[StateAttribute.Y_CENTER,0] = np.array([8,6])
        data[StateAttribute.X_CENTER,-1] = None
        data[StateAttribute.Y_CENTER,-1] = None
        data[StateAttribute.X_CENTER,-2] = np.array([8,6])
        data[StateAttribute.Y_CENTER,-2] = np.array([6,4])
        data[StateAttribute.X_CENTER,-3] = None
        data[StateAttribute.Y_CENTER,-3] = None
        data[StateAttribute.X_CENTER,-4] = np.array([0,0])
        data[StateAttribute.Y_CENTER,-4] = np.array([1,1])
        times_to_diff = [(-4,-2),(-2,0)]
        expected = {(StateAttribute.X_DISPLACEMENT,(-2,0)) : np.array([2,2]),
                    (StateAttribute.Y_DISPLACEMENT,(-2,0)) : np.array([2,2]),
                    (StateAttribute.X_DISPLACEMENT,(-4,-2)) : np.array([8,6]),
                    (StateAttribute.Y_DISPLACEMENT,(-4,-2)) : np.array([5,3])}
        result = my_models.transf_pos_to_displ(data,times_to_diff)
        self.assertTrue(all([np.all(result[k] == v) for k,v 
                                                    in expected.items()]))

class Test_Predictor_WAM(unittest.TestCase):

    @staticmethod
    def weights(Xbar,x,radius):
        Xbar = np.column_stack(list(Xbar.values())) # shape = (n_data,2)
        x = np.column_stack(list(x.values())) # shape = (1,2)?
        dists = np.sqrt(np.sum((Xbar-x)**2,axis=1))
        neighbors = dists <= radius
        weights = neighbors / neighbors.sum()
        return weights

    def test_compare_with_RadiusNeighborRegression(self):
        

        stateAttrs_in_codomain =  [StateAttribute.X_CENTER,
                                StateAttribute.Y_CENTER]  # the names of the attributes in the codomain is irrelevant
        attributes_in_balltree = [(StateAttribute.X_CENTER,0)
                                ,(StateAttribute.Y_CENTER,0)]

        attributes_in_domain = [(StateAttribute.X_CENTER,0)
                                ,(StateAttribute.Y_CENTER,0)]
        radius = 0.25 # I generate data in the unit square. 
        WAM = my_models.WAM_over_horizon(attributes_in_domain,
                            horizons = [1],
                            restrictions={},
                            weights=self.weights,
                            weights_parameters={'radius':radius},
                            radius = radius,
                            stateAttrs_in_codomain=stateAttrs_in_codomain,
                            attributes_in_balltree=attributes_in_balltree
                            )
        RNR = sklearn.neighbors.RadiusNeighborsRegressor(radius)
        
        n_data = 100
        inputs = np.random.uniform(size=(n_data,2))
        outputs = np.random.uniform(size=(n_data,2))
        RNR.fit(inputs,outputs)
        data = {(StateAttribute.X_CENTER,0): inputs[:,0]
                ,(StateAttribute.Y_CENTER,0): inputs[:,1]
                ,(StateAttribute.X_CENTER,1): outputs[:,0]
                ,(StateAttribute.Y_CENTER,1): outputs[:,1]}
        WAM.fit(data)
        test_inputs = np.random.uniform(size=(n_data,2))
        yhat_RNR = RNR.predict(test_inputs)
        test_inputs = {(StateAttribute.X_CENTER,0): test_inputs[:,0]
                        ,(StateAttribute.Y_CENTER,0): test_inputs[:,1]}
        yhat_WAM = WAM.predict(test_inputs)
        yhat_WAM = np.column_stack(list(yhat_WAM.values()))

        self.assertTrue(np.linalg.norm(yhat_RNR-yhat_WAM) < 1e-14)
        self.assertTrue(np.linalg.norm(yhat_WAM)>1e-10) # to catch if I did something strange

class Test_WAM_position(unittest.TestCase):

    def test_algorithm_with_two_trajs_two_samples_each(self):
        """
        Let the n:th sample on the i:th trajectory be xin. 
        Let a dash symbolize non-sampled parts of the trajectory.
        The data is visualized as:
                x11----x12

                x21----   --------x22
        This gives 1-step displacements visualized as
                d11---->
                d21---- -------->
        Let x be the test point and xhat be the corresponding predicted future 
        position. Then
                x11----x12
                x  ----   ----xhat
                x21----   ----    ----x22
        """
        stateAttrs_in_codomain = [StateAttribute.X_CENTER,
                                StateAttribute.Y_CENTER]
        stateAttrs_codomain_displ = [StateAttribute.X_DISPLACEMENT, # displacement dimensions must match the position dimensions
                                StateAttribute.Y_DISPLACEMENT]
        attributes_in_balltree = [(StateAttribute.X_CENTER,0)
                                ,(StateAttribute.Y_CENTER,0)]
        data = {(StateAttribute.X_CENTER,0) : np.array([0,0]), #(x11_x,x21_x)
                (StateAttribute.Y_CENTER,0) : np.array([2,0]), #(x11_y,x21_y)
                (StateAttribute.X_CENTER,1) : np.array([1,3]), #(x12_x,x22_x)
                (StateAttribute.Y_CENTER,1) : np.array([2,0]), #(x12_y,x22_y)
                (StateAttribute.SPEED,0) : np.array(2*[None])} # adding speed should be irrelevant
        data[StateAttribute.X_DISPLACEMENT,1] =\
            data[StateAttribute.X_CENTER,1]-data[StateAttribute.X_CENTER,0]
        data[StateAttribute.Y_DISPLACEMENT,1] = \
            data[StateAttribute.Y_CENTER,1]-data[StateAttribute.Y_CENTER,0]
        attributes_in_domain = [(StateAttribute.X_CENTER,0), 
                                (StateAttribute.Y_CENTER,0)]
        model = my_models.WAM_over_horizon_position(
                attributes_in_domain = attributes_in_domain,
                horizons = [1],
                restrictions = {},
                weights=functools.partial(my_models.weights,
                            diff = my_models.diff,
                            transformation = my_models.transf_identity),
                weights_parameters={'parameters':np.array([1])},
                radius=15
                ,stateAttrs_in_codomain=stateAttrs_in_codomain
                ,stateAttrs_codomain_displ=stateAttrs_codomain_displ
                ,attributes_in_balltree=attributes_in_balltree
                )
        model.fit(data)
        expected = {(StateAttribute.X_CENTER,1) : np.array([2]),
                    (StateAttribute.Y_CENTER,1) : np.array([1])}
        X = {(StateAttribute.X_CENTER,0) : np.array([0]),
                (StateAttribute.Y_CENTER,0) : np.array([1])}
        result = model.predict(X)
        self.assertTrue(all([np.all(result[k] == v) for k,v 
                                            in expected.items()]))


if __name__ == '__main__':
    unittest.main()

    