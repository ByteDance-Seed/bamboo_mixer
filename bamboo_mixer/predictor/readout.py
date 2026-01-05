import torch

from bamboo_mixer.utils import ActivationFunction


class ReadoutBlock(torch.nn.Module):

    def __init__(self, act: str = None, **config):
        super(ReadoutBlock, self).__init__()

        # Dynamically retrieve values from config
        self.act = ActivationFunction.get_activation(act)
        self.input_dim = config.get('input_dim')
        self.hidden_dims = config.get('hidden_dims', [])
        self.output_dim = config.get('output_dim')

        # Create the readout layer dynamically
        self.readout_block = self.create_readout_block()

    def create_readout_block(self):
        layers = []
        input_dim = self.input_dim

        # Create layers based on hidden dimensions
        for hidden_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            input_dim = hidden_dim  # Update input dim for the next layer

        # Add the final output layer
        layers.append(torch.nn.Linear(input_dim, self.output_dim))

        if self.act:
            layers.append(self.act)

        # Return the layers as a Sequential block
        return torch.nn.Sequential(*layers)

    def reset_parameters(self):
        for layer in self.readout_block:
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()

    def forward(self, x):
        return self.readout_block(x)
