from globalSettings import *

class AlexNetModel(pl.LightningModule):
    def __init__(self, aux_logits=False):
        super(AlexNetModel, self).__init__()
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 35, kernel_size=11, stride=4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.LocalResponseNorm(96),

            torch.nn.Conv2d(35, 384, kernel_size=5, stride = 2,padding = 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.LocalResponseNorm(384),

            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1,padding = 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(384),

            torch.nn.Conv2d(384, 384, kernel_size=3, stride=1,padding = 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(384),

            torch.nn.Conv2d(384, 256, kernel_size=3, stride=1,padding = 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(kernel_size=3, stride=2,padding = 1),
          )
        self.fc6 = torch.nn.Sequential(
            torch.nn.Linear(256,4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(4096),
          )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*4096,4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(4096, 8)
          )

    def forward_once(self, x):
        output= self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc6(output)
        return output


    def forward(self, uniform_patch, random_patch):
        output_fc6_uniform = self.forward_once(uniform_patch)
        output_fc6_random = self.forward_once(random_patch)
        output = torch.cat((output_fc6_uniform,output_fc6_random), 1)
        output = self.fc(output)
        return output, output_fc6_uniform, output_fc6_random
      
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=BaseModelConfig.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
       }

    def training_step(self, batch, batch_idx):
        uniform_patch, random_patch, random_patch_label = batch[0], batch[1], batch[2]
        output, _, _ = self(uniform_patch, random_patch)
        loss = torch.nn.functional.cross_entropy(output, random_patch_label)
        self.log("train_loss", loss.item())
        del uniform_patch, random_patch, random_patch_label
        return loss
    
    def validation_step(self, batch, batch_idx):
        uniform_patch, random_patch, random_patch_label = batch[0], batch[1], batch[2]
        output, _, _ = self(uniform_patch, random_patch)
        val_loss = torch.nn.functional.cross_entropy(output, random_patch_label)
        del uniform_patch, random_patch, random_patch_label
        self.log("val_loss", val_loss.item())

    def test_step(self, batch, batch_idx):
        uniform_patch, random_patch, random_patch_label = batch[0], batch[1], batch[2]
        output, _, _ = self(uniform_patch, random_patch)
        test_loss = torch.nn.functional.cross_entropy(output, random_patch_label)
        del uniform_patch, random_patch, random_patch_label
        self.log("test_loss", test_loss.item())


#class ResNetCifar10(pl.LightningModule):