import torch
from torch import nn

from .mask import Mask
from .graphwavenet import GraphWaveNet


class STDMAE(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmask_path,pre_trained_smask_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmask_path = pre_trained_tmask_path
        self.pre_trained_smask_path = pre_trained_smask_path
        # iniitalize 
        self.tmask = Mask(**mask_args)
        self.smask = Mask(**mask_args)

        self.backend = GraphWaveNet(**backend_args)

        # load pre-trained model
        self.load_pre_trained_model()


    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tmask_path)
        self.tmask.load_state_dict(checkpoint_dict["model_state_dict"])
        
        checkpoint_dict = torch.load(self.pre_trained_smask_path)
        self.smask.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmask.parameters():
            param.requires_grad = False
        for param in self.smask.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:
        """Feed forward of STMask.

        Args:
            history_data (torch.Tensor): Short-term historical data. shape: [B, L, N, 3]
            long_history_data (torch.Tensor): Long-term historical data. shape: [B, L * P, N, 3]
            future_data (torch.Tensor): future data
            batch_seen (int): number of batches that have been seen
            epoch (int): number of epochs
        Returns:
            torch.Tensor: prediction with shape [B, N, L].
            torch.Tensor: the Bernoulli distribution parameters with shape [B, N, N].
            torch.Tensor: the kNN graph with shape [B, N, N], which is used to guide the training of the dependency graph.
        """

        # reshape
        short_term_history = history_data     # [B, L, N, 1]
        long_term_history = long_history_data
        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = self.tmask(long_history_data[..., [0]])
        hidden_states_s = self.smask(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        
        # enhancing 
        out_len=1
        # hidden_states = hidden_states[:, :, -out_len:, :]# B,N,T,C=>B,N,96
        hidden_states = hidden_states[:, :, -out_len, :]

        y_hat = self.backend(short_term_history, hidden_states=hidden_states).transpose(1, 2).unsqueeze(-1)
        
        # hidden_states=hidden_states.view(batch_size,num_nodes,-1,1).transpose(1,2)


        #y_hat = self.backend(hidden_states)
        return y_hat

