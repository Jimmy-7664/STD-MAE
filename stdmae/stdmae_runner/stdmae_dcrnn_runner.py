from typing import Tuple, Union

import torch
import numpy as np

from basicts.runners import BaseTimeSeriesForecastingRunner
from basicts.metrics import masked_mae, masked_rmse, masked_mape


class STDMAEDCRNNRunner(BaseTimeSeriesForecastingRunner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

        # graph training


    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C]

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target feature

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value). [B, L, N, C] for each of them.
        """

        future_data, history_data, long_history_data = data

        history_data = self.to_running_device(history_data)  # B, L, N, C
        long_history_data = self.to_running_device(long_history_data)  # B, L, N, C
        future_data = self.to_running_device(future_data)  # B, L, N, C
        history_data = self.select_input_features(history_data)
        long_history_data = self.select_input_features(long_history_data)
        batch_size, seq_len, num_nodes, _ = future_data.shape
        if train:
            future_data_4_dec = self.select_input_features(future_data)
        else:
            future_data_4_dec = self.select_input_features(future_data)
            # only use the temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        prediction_data = self.model(history_data=history_data, long_history_data=long_history_data,
                                     future_data=future_data_4_dec,
                                     batch_seen=iter_num , epoch=epoch, train=train)

        assert list(prediction_data.shape)[:3] == [
            batch_size, seq_len,
            num_nodes], "error shape of the output, edit the forward function to reshape it to [B, L, N, C]"
        # post process
        prediction = self.select_target_features(prediction_data)
        real_value = self.select_target_features(future_data)
        return prediction, real_value


