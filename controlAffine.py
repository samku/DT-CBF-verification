import numpy as np

class controlAffine:
    def __init__(self):
        pass

    def f_x(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def g_x(self, x):
        raise NotImplementedError("This method should be overridden by subclasses")

    def K_x(self, x):
        #Ideal control input
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def BF_indicator(self, x):
        #Triplet of binariers = [h(x)<=0, h(x)>0, nabla h f+nabla h g u<=-alpha h]
        raise NotImplementedError("This method should be overridden by subclasses")