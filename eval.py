import torch
import torch.nn as nn
# import torch.optim as optim
import os
# import copy
import math
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
from qlib.config import REG_US, REG_CN
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from factormodel import FACTORMODEL
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# EPS = 1e-12


def inference(model, data_loader, stock2concept_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
          if stock2concept_matrix:
            pred,_ = model(feature, stock2concept_matrix[stock_index], market_value)
          else:
            pred,_ = model(feature, stock2concept_matrix, market_value)

            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args):

    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}
    segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder,segments)

    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    import pickle5 as pickle
    with open(args.market_value_path, "rb") as fh:
        df_market_value = pickle.load(fh)
    #df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    # print('!'*100)
    # print(slc)
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'], 
                   batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'], 
                              pin_memory=True, start_index=start_index, device = device)
    
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'], pin_memory=True, 
                             start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader

def main(args):
  
  
  print('create loaders...')
  train_loader, valid_loader, test_loader = create_loaders(args)
  
  stride = list(map(int,args.stride.split('+')))
  num_days = int(args.num_days)
  K = args.K
  
  model = FACTORMODEL(d_feat = args.d_feat, hidden_size = args.hidden_size, num_layers = args.num_layers, dilations = stride,num_days= num_days, K = K) 
  model.to(device)
  
  # print(model) 
  print('load model dict...')
  model.load_state_dict(torch.load(args.model_path))
  
  if K:
    stock2concept_matrix = np.load(args.stock2concept_matrix) 
    stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)
  else:
    stock2concept_matrix = None
  for name in ['train', 'valid', 'test']:
    pred = inference(model, eval(name+'_loader'), stock2concept_matrix)
    precision, recall, ic, rank_ic = metric_fn(pred)
    print(('%s: IC %.6f Rank IC %.6f')%(name, ic.mean(), rank_ic.mean()))
    print(name, ': Precision ', precision)
    print(name, ': Recall ', recall)


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

 
    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')
    parser.add_argument('--stride', type=str, default='1+2+5')
    parser.add_argument('--num_days', type=str, default='60')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')


    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')

    parser.add_argument('--model_path', default='./output/csi100_OursLR/model.bin.r0')

    args = parser.parse_args() #

    return args


if __name__ == '__main__':
    #
    args = parse_args()
    main(args)
