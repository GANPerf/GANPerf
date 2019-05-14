# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
import torch.nn as nn

from torchvision import models






class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        



        self.classifier = nn.Sequential(
            nn.Linear( 8, 128 ),
            #nn.Sigmoid(),
            nn.ReLU(True),
            nn.Dropout(),
           
           
            nn.Linear( 512, 512 ),
            #nn.Sigmoid(),
            nn.ReLU(True),
            nn.Dropout(),
          
      

            nn.Linear( 512, 1),
            nn.Sigmoid()
        )

        initialize_weights( self.classifier )


        

    def forward(self, x):
 

        return self.classifier(x)









class Model_D(nn.Module):
 
    def __init__(self):
        super(Model_D, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        initialize_weights( self.model )
        

    def forward(self, x):
        return self.model(x)





def initialize_weights( model ):

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.18)
            m.bias.data.zero_()





