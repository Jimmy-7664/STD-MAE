import torch
from torch import nn

from .mask import Mask
from .dcrnn import DCRNN


class STDMAE(nn.Module):
    """Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting"""

    def __init__(self, dataset_name, pre_trained_tmae_path,pre_trained_smae_path, mask_args, backend_args):
        super().__init__()
        self.dataset_name = dataset_name
        self.pre_trained_tmae_path = pre_trained_tmae_path
        self.pre_trained_smae_path = pre_trained_smae_path
        # iniitalize 
        self.tmae = Mask(**mask_args)
        self.smae = Mask(**mask_args)

        self.backend = DCRNN(**backend_args)

        # load pre-trained model
        self.load_pre_trained_model()


    def load_pre_trained_model(self):
        """Load pre-trained model"""

        # load parameters
        checkpoint_dict = torch.load(self.pre_trained_tmae_path)
        self.tmae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        checkpoint_dict = torch.load(self.pre_trained_smae_path)
        self.smae.load_state_dict(checkpoint_dict["model_state_dict"])
        
        # freeze parameters
        for param in self.tmae.parameters():
            param.requires_grad = False
        for param in self.smae.parameters():
            param.requires_grad = False
    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, **kwargs) -> torch.Tensor:

        # reshape
        short_term_history = history_data     # [B, L, N, 1]
        long_term_history = long_history_data
        batch_size, _, num_nodes, _ = history_data.shape

        hidden_states_t = self.tmae(long_history_data[..., [0]])
        hidden_states_s = self.smae(long_history_data[..., [0]])
        hidden_states=torch.cat((hidden_states_t,hidden_states_s),-1)
        
        # enhancing 
        out_len=1
        # hidden_states = hidden_states[:, :, -out_len:, :]# B,N,T,C=>B,N,96
        hidden_states = hidden_states[:, :, -out_len, :]

        y_hat = self.backend(history_data,future_data=future_data,batch_seen=batch_seen,hidden_states=hidden_states )
        
        # hidden_states=hidden_states.view(batch_size,num_nodes,-1,1).transpose(1,2)


        #y_hat = self.backend(hidden_states)
        return y_hat

