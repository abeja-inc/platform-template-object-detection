import os


def get_env_var(key, converter, default=None):
    value = os.environ.get(key)
    if value:
        return converter(value)
    return default


class Parameters:
    """Parameter class
    parameter name must be consist of upper case characters.
    """
    BATCH_SIZE = get_env_var('BATCH_SIZE', int, 16)
    EPOCHS = get_env_var('EPOCHS', int, 50)

    # NOTE: currently only SSD300 (size=300) is supported!
    IMG_SIZE = get_env_var('IMG_SIZE', int, 300)

    SHUFFLE = os.environ.get('SHUFFLE', 'True').lower() == 'true'
    RANDOM_SEED = get_env_var('RANDOM_SEED', int, None)

    # TODO: Load data on memory. If you use a big dataset, set it to `false`. Default `true`
    # USE_ON_MEMORY = os.environ.get('USE_ON_MEMORY', 'True').lower() == 'true'

    # TODO: Image cache. If you use a big dataset, set it to `false`.
    # If `USE_ON_MEMORY=true`, then `USE_CACHE=true` automatically. Default `true`
    # USE_CACHE = USE_ON_MEMORY or os.environ.get('USE_CACHE', 'True').lower() == 'true'

    MAX_ITEMS = get_env_var('MAX_ITEMS', int, None)

    TEST_SIZE = get_env_var('TEST_SIZE', float, 0.4)

    # TODO: early stopping
    # EARLY_STOPPING_TEST_SIZE = get_env_var('EARLY_STOPPING_TEST_SIZE', float, 0.2)
    # EARLY_STOPPING_PATIENCE = get_env_var('EARLY_STOPPING_PATIENCE', float, 5)

    # ----- SGD -----
    # cf. https://pytorch.org/docs/stable/optim.html#torch.optim.SGD
    LR = get_env_var('LEARNING_RATE', float, 1e-3)
    # momentum factor, if 0 means no momentum
    MOMENTUM = get_env_var('MOMENTUM', float, 0)
    # weight decay (L2 penalty)
    WEIGHT_DECAY = get_env_var('WEIGHT_DECAY', float, 0)
    # dampening for momentum
    DAMPENING = get_env_var('DAMPENING', float, 0)
    # Enables Nesterov momentum
    NESTEROV = os.environ.get('NESTEROV', 'False').lower() == 'true'

    # ----- Detect -----
    # Confidence threshold to filter out bounding boxes with low confidence
    CONF_THRESHOLD = get_env_var('CONF_THRESHOLD', float, 0.01)
    # Number of bounding boxes to be taken.
    TOP_K = get_env_var('TOP_K', int, 200)
    # The threshold for IoU to consider bounding boxes as the same
    NMS_THRESHOLD = get_env_var('NMS_THRESHOLD', float, 0.45)

    # ----- MultiBoxLoss -----
    # overlap threshold used when matching boxes.
    OVERLAP_THRESHOLD = get_env_var('OVERLAP_THRESHOLD', float, 0.5)
    # Hard Negative Mining ratio
    NEG_POS = get_env_var('NEG_POS', int, 3)

    # ----- PriorBox -----
    # variety of aspect ratio of output default box
    BBOX_ASPECT_NUM = [4, 6, 6, 6, 4, 4]
    # size of images of each source
    FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
    # decide size of default box
    STEPS = [8, 16, 32, 64, 100, 300]
    MIN_SIZES = [30, 60, 111, 162, 213, 264]
    MAX_SIZES = [60, 111, 162, 213, 264, 315]

    ASPECT_RATIOS = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    VARIANCE = [0.1, 0.2]
    CLIP = True
    # color mean (BGR)
    MEANS = (104, 117, 123)

    CLIP_VALUE = 0.2

    # ----- Inference -----
    CONFIDENCE_THRESHOLD = get_env_var('CONFIDENCE_THRESHOLD', float, 0.1)

    #
    ABEJA_TRAINING_RESULT_DIR = os.environ.get('ABEJA_TRAINING_RESULT_DIR', 'abejainc_training_result')

    @classmethod
    def as_dict(cls):
        return {
            k: v for k, v in cls.__dict__.items()
            if k.isupper()
        }
