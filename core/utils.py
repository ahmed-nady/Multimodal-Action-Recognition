# importing module
import logging
import time
import os
import random
import numpy as np
import torch

def getLogger(work_dir):

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # # Create and configure logger
    # logging.basicConfig(filename=f"{work_dir}/{timestamp}.log",
    #                     format='%(asctime)s %(message)s',
    #                     filemode='w')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(f"{work_dir}/{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    rootLogger = logging.getLogger(__name__)
    # logFormatter = logging.Formatter("%(asctime)s %(message)s")
    # # Creating an object
    # rootLogger = logging.getLogger()
    # fileHandler = logging.FileHandler(f"{work_dir}/{timestamp}.log")
    # fileHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(fileHandler)
    # rootLogger.setLevel(logging.INFO)
    # Setting the threshold of logger to DEBUG
    #logger.setLevel(logging.INFO)

    return rootLogger

#====mmaction2======
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
