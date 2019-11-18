import os

class Path(object):

    @staticmethod
    def db_root_dir(dataset):
        try:
            import nsml
            from nsml import HAS_DATASET, DATASET_PATH, DATASET_NAME
        except:
            DATASET_PATH = '/home/data/VOCseg_Aug2012/'

        if dataset == 'pascal':
            return os.path.join(DATASET_PATH, 'train/')  # folder that contains VOCdevkit/.
            # return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
