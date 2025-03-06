import numpy as np 

class Metrics:
      def compute(self, *args)->float:
            raise NotImplemented

class Accuracy(Metrics):
      def __init__(self):
            pass 
      def compute(self, pred_classes, true_classes):
            return (pred_classes == true_classes).sum()/len(pred_classes)*100