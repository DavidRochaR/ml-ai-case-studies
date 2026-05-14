"""Deep Learning fundamentals — MLP for MNIST."""
try:
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        """Multi-Layer Perceptron for MNIST."""
        def __init__(self, input_size=784, hidden=[256, 128], n_classes=10, dropout=0.2):
            super().__init__()
            layers = []
            prev = input_size
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.view(x.size(0), -1))
except ImportError:
    pass
