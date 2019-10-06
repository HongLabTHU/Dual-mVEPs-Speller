import argparse
import os
import json
from datetime import datetime
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from Offline import utils
from Offline.dataset import Dataset
from Offline.model import FeatExtractor
from Offline.model import Model
from config import cfg, merge_cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Offline training entry'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for subject',
        default=None,
        type=str
    )
    parser.add_argument(
        '--average-times',
        '-a',
        dest='n_avg',
        help='Times to take average',
        default=3,
        type=int
    )
    parser.add_argument(
        '--k-fold',
        '-k',
        dest='k',
        help='how many folds for cross validation',
        default=10,
        type=int
    )
    parser.add_argument(
        '--datetime',
        '-d',
        dest='date',
        help='dataset date index, default the folder with the latest date',
        default=None,
        type=str
    )
    parser.add_argument(
        '--parameter-search',
        '-p',
        dest='p',
        help='set to enable parameter search',
        action='store_true'
    )
    return parser.parse_args()


def _train(X_train, y_train, select, **kwargs):
    """

    :param X_train: epochs
    :param y_train: label
    :param select: selected channel index
    :return:
    model object
    """
    # create model object
    model = Model(subject=cfg.subj_info.subjname, **kwargs)
    model.ch_ind = select
    # select n channels
    train_x = X_train[:, model.ch_ind]
    # fit
    model.fit(train_x, y_train)
    return model


def _prediction(X, y, model, n_avg):
    """

    :param X: epochs
    :param y: label
    :param n_avg: average times
    :param model: trained model
    :return:
    """
    ind = model.ch_ind
    test_x = X[:, ind]
    # predict
    _y = utils.pred_ave(model, test_x, y, n=n_avg)
    return _y


def _k_fold(X, y, args, ch_select, **kwargs):
    """
    k fold validation function
    :param X: epochs
    :param y: label
    :param args:
    :param ch_select: channel selection result
    :return:
    figures: plt.figure
    result: dict, evaluation results
    ind: selected channels
    """
    y_pred = []
    y_true = []
    for ind_train, ind_test in utils.uniform_kfold(y, k=args.k, shuffle=True, random_state=52):
        # indexing in numpy array is deep copy
        X_train = X[ind_train]
        y_train = y[ind_train]
        X_test = X[ind_test]
        y_test = y[ind_test]
        # train model
        model = _train(X_train, y_train, select=ch_select, **kwargs)
        # test model
        _y = _prediction(X_test, y_test, model, 1)
        y_pred.append(_y)
        y_true.append(y_test)

    # evaluate model
    y_pred = np.concatenate(tuple(y_pred), axis=0)
    y_true = np.concatenate(tuple(y_true), axis=0)

    if cfg.exp_config.bidir:
        estimate_accu = utils.estimate_accu_dual(y_true=y_true, y_pred=y_pred, n_avg=args.n_avg)
        # taking the average
        y_pred = utils.average_multiclass(data=y_pred, labels=y_true, n=args.n_avg)
        results = utils.evaluate_multiclass(y_pred, y_true, if_plot=False)
    else:
        estimate_accu = utils.estimate_accu_uni(y_true=y_true, y_pred=y_pred, n_avg=args.n_avg)
        # taking the average
        y_pred = utils.average_multiclass(data=y_pred, labels=y_true, n=args.n_avg)
        results = utils.evaluate_binary(y_pred, y_true, if_plot=False)
    results['estimate_accu'] = estimate_accu
    return results


def _grid_search(X, y, args, ind, **kwargs):
    def product_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))

    results = _k_fold(X, y, args, ch_select=ind)
    # parameter search
    best_accuracy = np.sum(results['estimate_accu'])
    best_result = results
    best_params = None
    for params in product_dict(**kwargs):
        results = _k_fold(X, y, args, ch_select=ind, **params)
        if np.sum(results['estimate_accu']) > best_accuracy:
            best_params = params.copy()
            best_accuracy = np.sum(results['estimate_accu'])
            best_result = results
    return best_params, best_result


def main(args):
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    # loading dataset
    dataset = Dataset(subject=cfg.subj_info.subjname, date=args.date)
    # print dataset information
    print(dataset.data.shape)
    print(len(dataset.timestamp))
    print(dataset.events.shape)
    print(dataset.ch_names)
    # extract features
    extractor = FeatExtractor(sfreq=cfg.amp_info.samplerate,
                              band_erp=cfg.subj_info.erp_band)
    # select channels in montage
    data = dataset.data[dataset.montage_indices]
    ch_names = dataset.montage.copy()

    print(data.shape)

    model = Model(subject=cfg.subj_info.subjname)
    feature = model.extract_feature(extractor, data, channel_selection=False)
    X = model.raw2epoch(feature, dataset.timestamp, dataset.events)

    # get target
    if cfg.exp_config.bidir:
        target = utils.get_label_bidir(cfg.exp_config.train_string, cfg.exp_config.n_rep, dataset.events)
    else:
        target = utils.get_label(cfg.exp_config.train_string, cfg.exp_config.n_rep)
    y = target.flatten()

    try:
        plots_path = os.path.join(dataset.root_dir, 'plots')
        os.mkdir(plots_path)
    except FileExistsError:
        pass

    # split train/validate set and cross validating on train set
    # randomly split 20%
    train_ind, val_ind = utils.uniform_split(y, shuffle=True, random_state=42)
    X_train = X[train_ind]
    y_train = y[train_ind]
    X_val = X[val_ind]
    y_val = y[val_ind]
    ind = utils.chan_select(X_train, y_train, cfg.off_config.k_best_channel)
    print('Selected k best channel')
    print(ind)

    if args.p:
        if cfg.subj_info.type == 'eeg':
            # parameter search
            C = np.logspace(-4, 2, 10)
            n_components = range(1, (X_train.shape[1] // 2) + 1)
            best_params, best_result = _grid_search(X_train, y_train, args, ind, C=C, n_components=n_components)
        else:
            raise KeyError('Unsupported experiment type.')
        for key in best_params:
            print('Best %s: %.4f' % (key, best_params[key]))
        for i in best_result:
            print('%s: ' % i, best_result[i])
        print('')
    else:
        if cfg.subj_info.type == 'eeg':
            best_params = {'C': 1., 'n_components': 3}
        else:
            raise KeyError('Unsupported experiment type.')
        best_result = _k_fold(X_train, y_train, args, ch_select=ind)

    # train a model with selected parameters
    model = _train(X_train, y_train, select=ind, **best_params)
    y_pred = _prediction(X_val, y_val, model, args.n_avg)

    if cfg.exp_config.bidir:
        fig_roc, fig_cm, fig_pr, param_dict = utils.evaluate_multiclass(y_pred, y_val, if_plot=True)
    else:
        fig_roc, fig_cm, fig_pr, param_dict = utils.evaluate_binary(y_pred, y_val, if_plot=True)

    print('Cross validation results:')
    for i in best_result:
        print('%s: ' % i, best_result[i])
    print('')
    print('Split out validation result:')
    for i in param_dict:
        print('%s: %.4f' % (i, param_dict[i]))
    print('')
    # save evaluation results
    fig_roc.savefig(os.path.join(plots_path, 'roc.png'))
    fig_cm.savefig(os.path.join(plots_path, 'cm.png'))
    fig_pr.savefig(os.path.join(plots_path, 'pr.png'))
    # dump training info
    info = {
        'average': args.n_avg,
        'k-fold': args.k,
        'selected channels': [ch_names[i] + ' ERPs' if i < len(ch_names) else ch_names[i - len(ch_names)] + ' HG' for i
                              in ind],
    }
    info.update(best_params)
    try:
        os.mkdir(os.path.join(dataset.root_dir, 'logs'))
    except FileExistsError:
        pass
    with open(os.path.join(dataset.root_dir, 'logs', 'log%s.txt' % datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'w') as f:
        # training parameters
        f.write(json.dumps(info))
        f.write('\n')
        f.write('Cross validation result:\n')
        f.write(json.dumps(best_result))
        f.write('\n')
        f.write('Split out validation result:\n')
        f.write(json.dumps(param_dict))
        f.write('\n')
        f.write('\n')
        f.write(json.dumps(cfg.subj_info))
        f.write('\n')
        f.write('\n')
        f.write(json.dumps(cfg.off_config))

    # training with whole dataset and save model
    model = _train(X, y, select=ind, date=args.date, **best_params)
    model.dump()
    # show plots
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
