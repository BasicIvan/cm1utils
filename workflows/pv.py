class peakConcentration:
    def __init__(self,in_dict):
        self.my_dict = in_dict

    def getConcentration(self,xpos:str)->float:
        return self.my_dict[xpos][0]

    def getPeakRatio(self,nomXpos:str,denomXpos:str)->float:
        """Returns: ratio of peak concentrations of the two xpos (float)"""
        return self.my_dict[nomXpos][0]/self.my_dict[denomXpos][0]
    
    def getTimingDiff(self,nomXpos:str,denomXpos:str)->int:
        """Returns: difference in time indices between the two xpos (int).
        The difference should be multiplied by the model output time, 
        which is 10 minutes in these simulations"""
        return self.my_dict[nomXpos][1]-self.my_dict[denomXpos][1]