from .synthetic_dataset import SyntheticDataSet


DATA_SETS = {
    'synthetic': SyntheticDataSet
}


def get_dataset(config):
    try:
        ds_name = config.name

        if ds_name in DATA_SETS:
            data_set = DATA_SETS[ds_name](config)
        else:
            raise NotImplementedError(f'Not implemented dataset: {ds_name}')

    except Exception as e:
        raise Exception(f'get_dataset failed: {repr(e)}')

    return data_set
