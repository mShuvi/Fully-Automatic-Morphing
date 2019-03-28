from os import makedirs, rename, listdir, getcwd
from os.path import exists, isfile, isdir, join
from datetime import datetime
from json import loads
from collections import OrderedDict
from argparse import Namespace


class StopTrain:
    def __init__(self, delta=1e-6, patience=200, coldstart=20):
        self.delta = delta
        self.patience = patience
        self.counter = 0
        self.prev_loss = float('inf')
        self.coldstart = coldstart
        self.coldstart_count = 0

    def step(self, loss):
        if self.coldstart_count < self.coldstart:
            self.coldstart_count += 1
        else:
            if self.prev_loss - loss < self.delta:
                self.counter += 1
            else:
                self.counter = 0
            self.prev_loss = loss
            if self.counter == self.patience:
                return True
        return False

    def reset(self):
        self.counter = 0
        self.coldstart_count = 0


def get_timestamp():
    return datetime.now().strftime('%d.%m.20%y_%H:%M:%S')


def mkdir(path, use_timestamp=True):
    full_path = "%s" % path
    if use_timestamp:
        full_path += "_%s" % get_timestamp()
    if not exists(full_path):
        makedirs(full_path)


def mkdirs(paths, use_timestamp=True):
    if isinstance(paths, str):
        mkdir(paths, use_timestamp)
    else:
        for path in paths:
            mkdir(path, use_timestamp)


def mkdir_and_rename(path):
    if exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        rename(path, new_name)
    mkdir(path, use_timestamp=False)


def get_options_dict(opt_path):
    def dict_to_namespace(d):
        for k in d.keys():
            if type(d[k]) is OrderedDict:
                d[k] = dict_to_namespace(d[k])
        return Namespace(**d)

    if isdir(opt_path):
        opt_list = [join(opt_path, x) for x in listdir(opt_path) if x.lower().endswith('json')]
        opt_list.sort()
        if len(opt_list) == 0:
            raise ValueError('Given directory does not contain any files ending with .json.')
    elif isfile(opt_path) and opt_path.lower().endswith('json'):
        opt_list = [opt_path]
    else:
        raise TypeError('Given path is neither a directory nor a file ending with .json.')

    namespace_list = []
    for opt in opt_list:
        json_str = ''
        with open(opt, 'r') as f:
            for line in f:
                line = line.split('//')[0] + '\n'
                json_str += line
        d = loads(json_str, object_pairs_hook=OrderedDict)
        ns = dict_to_namespace(d)
        ns.__setattr__('opt_path', opt)
        namespace_list.append(ns)
    return namespace_list


def namespace_to_dict(ns):
    d = vars(ns)
    for k in d.keys():
        if type(d[k]) is Namespace:
            d[k] = namespace_to_dict(d[k])
    return d


def str_to_list(string, delimiter=',', coef=1, to_int=False):
    dtype = int if to_int else float
    if type(string) is list:
        string = str(string)[1:-1]
    lst = string.split(delimiter)
    for i in range(len(lst)):
        lst[i] = dtype(lst[i]) * coef
    return lst
