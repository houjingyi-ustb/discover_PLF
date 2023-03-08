import torch.nn as nn
import torch.nn.init as init


class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.D = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        for block in self._modules:
            for m in self._modules[block]:
              if isinstance(m, nn.Linear):
                init.normal_(m.weight, 0, 0.02)
                m.bias.data.fill_(0)
        

    def forward(self, z):
        return self.D(z).squeeze()

