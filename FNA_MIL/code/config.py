import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


def get_config_str(args):
    parser = get_parser()
    # no args to get default
    config, unparsed = parser.parse_known_args(args=args.strip().split(' '))
    return config


def get_parser():
    parser = argparse.ArgumentParser(description='covid')

    # data params
    data_arg = parser.add_argument_group('Data Params')
    data_arg.add_argument('--batch_size', type=int, default=1,
                          help='# of images in each batch of data')
    data_arg.add_argument('--center_crop', type=int, default=224,
                          help='Size of the initial center crop to use')
    data_arg.add_argument('--shear', type=int, default=0,
                          help='Amount of random shear to apply')
    data_arg.add_argument('--speckle', type=float, default=0,
                          help='Amount of speckle noise to add')
    data_arg.add_argument('--saturation', type=float, default=0,
                          help='Amount of extra saturation jittering')
    data_arg.add_argument('--hue', type=float, default=0,
                          help='Amount of extra hue jittering')
    data_arg.add_argument('--test_data_index', type=int, default=-1,
                          help='Which data will be used for testing for leave one out')
    data_arg.add_argument('--test_data_fold', type=int, default=-1,
                          help='Which data will be used for testing for K-fold')

    train_arg = parser.add_argument_group('Training Params')
    train_arg.add_argument('--epochs', type=int, default=50,
                           help='# of epochs to train for')
    train_arg.add_argument('--init_lr', type=float, default=1e-3,
                           help='Initial learning rate value')
    train_arg.add_argument('--lr_patience', type=int, default=5,
                           help='Number of epochs to wait before reducing lr')
    train_arg.add_argument('--train_patience', type=int, default=1000,
                           help='Number of epochs to wait before stopping train')
    train_arg.add_argument('--random_seed', type=int, default=0,
                           help='Random seed')
    train_arg.add_argument('--model_name', type=str, default='densenet',
                           help='which architecture to use')
    train_arg.add_argument('--pretrained_model', type=str2bool, default=True,
                           help='pretraining True or False')
    train_arg.add_argument('--optimizer', type=str, default='adamw')

    misc_arg = parser.add_argument_group('Misc.')
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                          help="Whether to run on the GPU")
    misc_arg.add_argument('--gpu_number', type=int, default=3,
                          help="Which GPU to use")
    misc_arg.add_argument('--mil_size', type=int, default=8)
    misc_arg.add_argument('--test_interval', type=int, default=5,
                          help="How often to evaluate the test set")
    misc_arg.add_argument('--experiment_name', type=str, default='default',
                          help='Name of experiment')
    misc_arg.add_argument('--lq_loss', type=float, default=None)
    misc_arg.add_argument('--lr_schedule', type=str, default=None,
                          help='Type of lr schedule to use')
    misc_arg.add_argument('--model_id', type=str, default=None,
                          help='MIL only: model to use for the warm start')
    return parser


def get_config():
    parser = get_parser()
    config, unparsed = parser.parse_known_args()
    return config, unparsed