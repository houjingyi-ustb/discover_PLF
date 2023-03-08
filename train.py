import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
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
from torch.utils.tensorboard import SummaryWriter

from factormodel import FACTORMODEL
from utils import metric_fn, mse, cal_beta
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])


global_log_file = None
def pprint(*args):
    # print with PDT time
    time = '['+str(datetime.datetime.utcnow()-
                   datetime.timedelta(hours=7))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix = None):

    global global_step

    model.train()
    
    if args.beta_strategy == 'B':
        C_max = torch.FloatTensor([0.50]).to(device)
        C = torch.clamp(C_max/args.n_epochs*epoch, 0, C_max.data[0])

    for _, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        # print(slc)
        feature, label, market_value , stock_index, _ = train_loader.get(slc)
        # print(feature.size())
        loss_vae = 0.
        pred, loss_vae = model(feature, stock2concept_matrix[stock_index], market_value)
        loss_pred = loss_fn(pred, label, args)

        if args.beta_strategy == 'B':
          loss = loss_pred+(loss_vae[0]+loss_vae[2]+(loss_vae[1]-C).abs())*5e-2
        elif args.beta_strategy == 'increasing':
          beta = cal_beta(epoch)
          loss = loss_pred+(loss_vae[0]+beta*loss_vae[1]+loss_vae[2])*5e-2
        else:
          raise ValueError('unknown strategy name `%s`'%args.beta_strategy)
          
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)

        with torch.no_grad():
            pred, _ = model(feature, stock2concept_matrix[stock_index], market_value)
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic


    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic

def inference(model, data_loader, stock2concept_matrix=None):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            pred,_ = model(feature, stock2concept_matrix[stock_index], market_value)

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
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)


    if args.K:
      model_name = 'OursHIST'
    else:
      model_name = 'OursLR'
    # outdir = args.outdir + args.data_set
    output_path = args.outdir + args.data_set + '_' + model_name

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file

    global_log_file = output_path + '/' + args.data_set + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    stock2concept_matrix = np.load(args.stock2concept_matrix) 
    stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)


    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(args.repeat):
        pprint('create model...')
        stride = list(map(int,args.stride.split('+')))
        num_days = int(args.num_days)
        model = FACTORMODEL(d_feat = args.d_feat, hidden_size = args.hidden_size, num_layers = args.num_layers, dilations = stride, num_days= num_days, K = args.K)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)
        for epoch in range(args.n_epochs):
            pprint('Running', times,'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix)


            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            
            if epoch<20: continue 
            
            model.load_state_dict(avg_params)

            pprint('evaluating...')

            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, stock2concept_matrix, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix, prefix='Test')

            pprint('valid_loss %.6f, test_loss %.6f'%(val_loss, test_loss))

            pprint('valid_ic %.6f, test_ic %.6f'%(val_ic, test_ic))

            pprint('valid_rank_ic %.6f, test_rank_ic %.6f'%(val_rank_ic, test_rank_ic))
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            model.load_state_dict(params_ckpt)
            
            if val_score > best_score:
                best_score = val_score
                stop_round = 0 
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break
            pprint('stop_round: ', stop_round)

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin.r'+str(times))

        pprint('inference...')
        
        
        res = dict()
        for name in ['train', 'valid', 'test']:

            pred= inference(model, eval(name+'_loader'), stock2concept_matrix)
            pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            precision, recall, ic, rank_ic = metric_fn(pred)

            pprint(('%s: IC %.6f Rank IC %.6f')%(
                        name, ic.mean(), rank_ic.mean()))
            pprint(name, ': Precision ', precision)
            pprint(name, ': Recall ', recall)
            res[name+'-IC'] = ic
            # res[name+'-ICIR'] = ic.mean() / ic.std()
            res[name+'-RankIC'] = rank_ic
            # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()
        
        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis= 0)
    precision_std = np.array(all_precision).std(axis= 0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')


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

    # training
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=25)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=10)
    parser.add_argument('--beta_strategy', default='increasing')

 
    # data
    parser.add_argument('--data_set', type=str, default='csi100')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicates daily batch
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


    # input for HIST
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')

    parser.add_argument('--outdir', default='./output/')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args() #

    return args


if __name__ == '__main__':
    #
    args = parse_args()
    main(args)
