#!/bin/bash
python scripts/data_preparation/PEMS03/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS07/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS08/generate_training_data.py --history_seq_len 12

python scripts/data_preparation/PEMS03/generate_training_data.py --history_seq_len 864
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 864
python scripts/data_preparation/PEMS07/generate_training_data.py --history_seq_len 864
python scripts/data_preparation/PEMS08/generate_training_data.py --history_seq_len 2016