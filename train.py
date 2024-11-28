import torch
import multiprocessing
import traceback
import os, sys
from math import ceil
from time import time
from sklearn.metrics import accuracy_score
from datapreprocessing import get_dataloader
from EBSNN_LSTM import EBSNN_LSTM
from EBSNN_GRU import EBSNN_GRU
import argparse

base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.join(base_path),os.path.join(base_path, "../"),base_path.rsplit('/')[0]])
from utils import p_log
from utils import deal_results
from utils import set_log_file
from evaluate import evaluate_loss_acc
from evaluate import FocalLoss

from tensorboardX import SummaryWriter
# Currently I cannot find the pytorchtools...
# from pytorchtools import EarlyStopping


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument(
    '--model', default='EBSNN_LSTM',
    help='Model name: EBSNN_LSTM or EBSNN_GRU [default: EBSNN_LSTM]')
parser.add_argument(
    '--batch_size', type=int, default=32,
    help='Batch size [default: 32]')
parser.add_argument('--epochs', type=int, default=50,
                    help='Epochs [default: 50]')
parser.add_argument(
    '--flow', action='store_true',
    help='if flow classification?')
parser.add_argument(
    '--gamma', type=float, default=2,
    help='gamma for focal loss [default 2]')
parser.add_argument('--test_percent', type=float, default=0.2,
                    help='test percent [default 0.2]')
parser.add_argument('--embedding_dim', type=int, default=257,
                    help='embedding dimenstion [default 257]')
parser.add_argument(
    '--filename', type=str,
    default='result_flow_threshold_3_class_12.traffic',
    help='file name of input dataset')
parser.add_argument(
    '--log_filename', type=str,
    default='log_20/log_train.txt',
    help='file name of log'
)
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate [default 0.001]')
# parser.add_argument('--patience', type=int, default=100,
#                     help='patience epochs for early_stopping [default 100]')
parser.add_argument(
    '--labels', type=str,
    default='skype,pplive,baidu,tudou,weibo,thunder,youku,itunes,'
    'taobao,qq,gmail,sohu',
    help='names of labels, seperated by ",", modify it if you need')
# top_k, aggregate stragety
parser.add_argument(
    '--first_k_packets', type=int, default=3,
    help='first_k_packets for flow classification, value must in '
    'range [1, threshold] [default 3]')
parser.add_argument(
    '--aggregate', type=str, default='sum_max',
    help='aggregate stragety for flow classification, sum_max or '
    'count_max [default sum_max]')
parser.add_argument('--debug', action='store_true', help='debug')
parser.add_argument('--shuffle', action='store_true', help='if shuffle dataset')
parser.add_argument('--no_bidirectional', action='store_true',
                    help='if bi-RNN')
parser.add_argument('--segment_len', type=int, default=8,
                    help='the length of segment')
parser.add_argument('--test_cycle', type=int, default=1,
                    help='test cycle')
FLAGS = parser.parse_args()

if __name__ == '__main__':
    timer_start = time()
    DEVICE = FLAGS.gpu
    MODEL = {
        'EBSNN_LSTM': EBSNN_LSTM, 'EBSNN_GRU': EBSNN_GRU
    }[FLAGS.model]
    BATCH_SIZE = FLAGS.batch_size
    LOG_FILENAME = FLAGS.log_filename
    set_log_file(LOG_FILENAME)
    p_log(f'start preparing for training, log_filename={LOG_FILENAME}')
    EPOCHS = FLAGS.epochs
    FLOW = FLAGS.flow
    DEBUG = FLAGS.debug
    SHUFFLE = FLAGS.shuffle
    GAMMA = FLAGS.gamma
    SEGMENT_LEN = FLAGS.segment_len
    test_cycle = FLAGS.test_cycle
    test_percent = FLAGS.test_percent
    # patience = FLAGS.patience
    EMBEDDING_DIM = FLAGS.embedding_dim
    BIDIRECTION = not FLAGS.no_bidirectional
    LR = FLAGS.learning_rate
    FILENAME = FLAGS.filename
    LABELS = {v: k for k, v in enumerate(FLAGS.labels.split(','))}
    NUM_CLASS = len(LABELS)
    p_log('LABELS: {}'.format(LABELS))
    #!! TODO: check validity for FIRST_K_PKGS
    FIRST_K_PKGS = FLAGS.first_k_packets
    AGGREGATE = FLAGS.aggregate
    valid_aggregate = ['sum_max', 'count_max']
    if AGGREGATE not in valid_aggregate:
        p_log('unexpected values of aggregate: {}, expected:'
              ' value in {}'.format(AGGREGATE, valid_aggregate))
        sys.exit(-1)

    # Tensorboard
    log_writer = SummaryWriter()

    # data
    train_loader, test_loader = get_dataloader(
        FILENAME, LABELS, test_percent, BATCH_SIZE,
        flow=FLOW, first_k_packets=FIRST_K_PKGS,
        segment_len=SEGMENT_LEN, shuffle=SHUFFLE)
    # debug
    # _, debug_test_loader = get_dataloader(
    #     'newtest.traffic', LABELS, test_percent=1.0,
    #     batch_size=BATCH_SIZE, flow=FLOW,
    #     first_k_packets=FIRST_K_PKGS,
    #     segment_len=SEGMENT_LEN, shuffle=SHUFFLE)
    # model
    model = MODEL(NUM_CLASS, EMBEDDING_DIM, DEVICE,
                  segment_len=SEGMENT_LEN,
                  bidirectional=BIDIRECTION)

    # model.load_state_dict(torch.load(load_model_name))
    # overall_label_ix = (torch.arange(0, NUM_CLASS)).long()
    model = model.cuda(DEVICE)
    # overall_label_ix = overall_label_ix.cuda(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # loss_func = nn.CrossEntropyLoss()
    loss_func = FocalLoss(
        NUM_CLASS, DEVICE, alpha=train_loader.alpha,
        gamma=GAMMA, size_average=True)

    num_batch = len(train_loader)
    # patience is how long to wait after last time validation loss improved.
    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_results = {'acc': 0., 'epoch': 0, 'results': None}
    output_after_batches = ceil((num_batch / 4) / 100) * 100
    cpu_cnt = multiprocessing.cpu_count()
    # train
    for epoch in range(EPOCHS):
        # Avoid extremely poor resource situation, i.e. load average far
        # more than cpu core count
        load_avg = os.getloadavg()[1]
        # while load_avg > 3 * cpu_cnt:
        #     # wait for 20min
        #     p_log('Current laod average is very high! '
        #           '({} while cpu cores={})'.format(load_avg, cpu_cnt))
        #     time.sleep(20 * 60)
        train_loss = 0.
        train_acc_avg = 0.
        y = []
        y_hat = []
        s_t = time()
        for i in range(num_batch):
            batch_X, batch_y = train_loader[i]
            batch_X = batch_X.cuda(DEVICE)
            y += batch_y.tolist()
            batch_y = batch_y.cuda(DEVICE)

            # begin to train
            model.train()
            optimizer.zero_grad()
            out = model(batch_X)
            y_hat += out.max(1)[1].tolist()
            train_acc_avg += accuracy_score(
                batch_y.tolist(), out.max(1)[1].tolist())
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / num_batch
        train_acc_avg = train_acc_avg / num_batch
        # train_accuracy = accuracy_score(y, y_hat)
    # save model
    save_model_name = './models/saved_final_checkpoint.pt'
    torch.save(model.state_dict(), save_model_name)
    model_state, test_acc, epoch = model_best
    save_model_name = './models/best_ckpt_epoch{}_testacc{:.4f}.pt'.format(
        epoch, test_acc
    )
    torch.save(model_state, save_model_name)
    log_writer.add_graph(model, input_to_model=batch_X)
    # export scalar data to JSON for external processing
    log_writer.export_scalars_to_json("./all_scalars.json")
    log_writer.close()
