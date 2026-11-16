# imports to recreate model predictions
import torch
import torch.nn as nn

# model architecture
class sat_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),


            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

        )
        
        self.pool = nn.AdaptiveAvgPool2d((1,1))                  
    
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(512, 256),                         
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(256, 10),                               
        )
    
    def forward(self,x):
        x = self.conv_stack(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

# map to convert numeric into nominal labels    
inv_map = {
    0: 'HerbaceousVegetation',
    1: 'AnnualCrop',
    2: 'Residential',
    3: 'Pasture',
    4: 'Industrial',
    5: 'River',
    6: 'Highway',
    7: 'Forest',
    8: 'PermanentCrop',
    9: 'SeaLake'
}

# transform(normalization) values
mean = [0.34368882, 0.38104076, 0.40845288]
std = [0.20109171, 0.13641693, 0.11618139]