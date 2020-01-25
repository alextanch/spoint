import yaml
import logging
import argparse

from tqdm import tqdm

from spoint.utils.config import Config
from spoint.datasets.loader import Loader


def train(epoch, loader):
    epochs = loader.epochs

    for batch in tqdm(loader, desc=f'train epoch {epoch}/{epochs}', unit='batch', ascii=True):
        continue


def main(config):
    logging.info('running train')

    logging.info('get loader')
    loader = Loader(config['loader'])

    for epoch in range(loader.start_epoch, loader.epochs):
        train(epoch, loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('cfg', type=str, help='config file')
    parser.add_argument('--debug', action='store_true', default=False)

    config = Config(parser.parse_args().__dict__)

    with open(config.cfg, 'r') as fp:
        config.update(Config(yaml.load(fp, Loader=yaml.FullLoader)))

    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG if config.debug else logging.INFO)

    main(config)
