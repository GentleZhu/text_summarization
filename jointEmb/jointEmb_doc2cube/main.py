from graph import HinBuilder, HinLoader
from IPython import embed
import argparse
import numpy as np
import os
from emb_lib import SkipGram
import torch as t
import pickle
import json
import torch.utils.data as tdata
from WikidataLinker_mg import WikidataLinker

def parse_args():
    '''
    Parses the heer arguments.
    '''
    parser = argparse.ArgumentParser(description="Run heer.")

    parser.add_argument('--more-param', nargs='?', default='None',
                        help='customized parameter setting')

    parser.add_argument('--input', nargs='?', default='data/edge.txt',
                        help='Input graph path')

    parser.add_argument('--gpu', nargs='?', default='0',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size. Default is 50.')

    parser.add_argument('--window-size', type=int, default=1,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--pre-train-path', type=str, default='',
                        help='embedding initialization')
    parser.add_argument('--pre-load-model', type=str, default=None,
                        help='module initialization')

    parser.add_argument('--build-graph', type=bool, default=True,
                        help='heterogeneous information network construction')

    parser.add_argument('--graph-name', type=str, default='doc2cube',
                        help='prefix of dumped data')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='data directory')
    parser.add_argument('--model-dir', type=str, default='model/',
                        help='model directory')
    parser.add_argument('--log-dir', type=str, default='log/',
                        help='log directory')
    parser.add_argument('--fine-tune', type=int, default=0,
                        help='fine tune phase')

    parser.add_argument('--iter', default=40, type=int,
                        help='Number of epochs in SGD')
    parser.add_argument('--op', default=-1, type=int)
    parser.add_argument('--map_func', default=-1, type=int)
    parser.add_argument('--dump-timer', default=5, type=int)

    parser.add_argument('--weighted', default=True)
    parser.add_argument('--unweighted', default=False)

    return parser.parse_args()


def learn_embeddings():
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    print('Network Spec:', config)

    # flexible param interface for tuning
    more_param = args.more_param  # everything separated by underscore, e.g., rescale_0.1_lr_0.02
    more_param_dict = {}  # {param_key: param_value_str}
    if more_param != 'None':
        more_param_list = more_param.split("_")
        assert len(more_param_list) % 2 == 0
        for i in xrange(0, len(more_param_list), 2):
            more_param_dict[more_param_list[i]] = more_param_list[i + 1]
    rescale_factor = 1. if 'rescale' not in more_param_dict else float(more_param_dict['rescale'])
    learning_rate = 1. if 'lr' not in more_param_dict else float(
        more_param_dict['lr'])  # please keep default values consistent with records on our google spreadsheet
    learning_rate_ratio = 16. if 'lrr' not in more_param_dict else float(
        more_param_dict['lrr'])  # please keep default values consistent with records on our google spreadsheet

    _data = ''
    if len(args.pre_train_path) > 0:
        _data = rescale_factor * utils.load_emb(args.data_dir, args.pre_train_path, args.dimensions, args.graph_name,
                                                config['nodes'])
    if args.weighted:
        _network = tdata.TensorDataset(t.LongTensor(pickle.load(open(args.data_dir + args.graph_name + '_input.p', 'rb'))),
                                       t.LongTensor(pickle.load(open(args.data_dir + args.graph_name + '_output.p', 'rb'))),
                                       t.LongTensor(pickle.load(open(args.data_dir + args.graph_name + '_weight.p', 'rb'))))
    else:
        _network = tdata.TensorDataset(t.LongTensor(pickle.load(open(args.data_dir + args.graph_name + '_input.p', 'rb'))),
                                       t.LongTensor(pickle.load(open(args.data_dir + args.graph_name + '_output.p', 'rb'))))
    model = SkipGram({'emb_size': args.dimensions, 'weighted': args.weighted,
                      'window_size': 1, 'batch_size': args.batch_size, 'iter': args.iter, 'neg_ratio': 5,
                      'graph_name': args.graph_name, 'dump_timer': args.dump_timer, 'model_dir': args.model_dir,
                      'log_dir': args.log_dir,
                      'data_dir': args.data_dir, 'mode': args.op, 'map_mode': args.map_func,
                      'fine_tune': args.fine_tune,
                      'lr_ratio': learning_rate_ratio, 'lr': learning_rate, 'network': _network,
                      'more_param': args.more_param,
                      'pre_train': _data, 'node_types': config['nodes'], 'edge_types': config['edges']})

    if args.pre_load_model:
        pre_load_model = t.load(args.pre_load_model, map_location=lambda storage, loc: storage)
        model.neg_loss.load_state_dict(pre_load_model)
        model.cuda()

    model.train()
    embedding = model.output()
    embed()
    exit()
    return

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    global config
    #config_name = os.path.join(args.data_dir.replace('intermediate', 'input'), args.graph_name.split('_ko_')[0] + '.config')
    #config = utils.read_config(config_name)
    config = {'edges': [[0, 1, 0], [2, 1, 1]], 'nodes': ['D', 'P', 'L'], 'types': ['DP', 'LP']}
    if args.build_graph:
        #print(args.node_types)
        tmp = HinLoader({'graph': args.input, 'types':config['nodes'], 'edge_types':config['edges']},  weighted = True)
        tmp.readHin(config['types'])
        tmp.encode()
        tmp.dump(args.data_dir + args.graph_name)
        #print(args.edge_types)
    else:
        learn_embeddings()

if __name__ == "__main__":
    args = parse_args()
    #read_hin(args.input)
    #t.cuda.set_device(int(args.gpu))
    main(args)
