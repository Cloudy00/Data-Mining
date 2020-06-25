import os
from argparse import ArgumentParser
from configparser import ConfigParser
import torch


class Config(ConfigParser):
    batch_size = 32
    data_path ="tnews_public"
    save_vocab ="results/vocab"
    save_model ="results/model"
    save_log = "text_cnn.log"
    min_freq = 5
    embed_dim = 256
    max_seq_len = 100
    num_kernel = 16
    kernel_sizes = [2, 3, 4]
    num_class = 15
    num_channel = 1
    hidden_size = 256
    dropout = 0.5
    epochs = 10
    lr = 5e-4


    def __init__(self, config_file):
        raw_config = ConfigParser()
        raw_config.read(config_file)
        self.cast_values(raw_config)

    def cast_values(self, raw_config):
        for section in raw_config.sections():
            for key, value in raw_config.items(section):
                val = None
                if type(value) is str and value.startswith("[") and value.endswith("]"):
                    val = eval(value)
                    setattr(self, key, val)
                    continue
                for attr in ["getint", "getfloat", "getboolean"]:
                    try:
                        val = getattr(raw_config[section], attr)(key)
                        break
                    except:
                        val = value
                setattr(self, key, val)


def parse_config():
    parser = ArgumentParser(
        description="Text CNN")
    parser.add_argument('--config', dest='config', default='CONFIG')
    # action='store_true') # for debug
    parser.add_argument('--train', dest="train", default=True)
    # action='store_true') # for debug
    parser.add_argument('--test', dest="test", default=True)
    parser.add_argument('-v', '--verbose', default=False)

    args = parser.parse_args()
    config = Config(args.config)

    config.train = args.train
    config.test = args.test
    config.verbose = args.verbose
    config.device = torch.device(
        "cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    return config
