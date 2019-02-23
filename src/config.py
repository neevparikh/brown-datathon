""" Config class for search/augment """
import argparse
import os

class Config(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text
    def build_parser(self):
        parser = argparse.ArgumentParser('scene')
        parser.add_argument('--name', required=True)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--print_freq', type=int, default=50,
                help='print frequency')
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--epochs', type=int, default=50,
                help='# of training epochs')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--first_prop', type=float, default=.4, 
                help='first part of lr cycle')
        parser.add_argument('--w_lr_start', type=float, default=3e-4,
                help='start lr for weights')
        parser.add_argument('--w_lr_middle', type=float, default=3e-3,
                help='middle lr for weights')
        parser.add_argument('--w_lr_end', type=float, default=3e-5,
                help='end lr for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-1)
        parser.add_argument('--w_weight_decay_end', type=float, default=3e-4,
                help='final weight decay, only used for wd finder')
        parser.add_argument('--total_channels_to_add', type=int, default=100,
                help='# of channels to add to net in total')
        parser.add_argument('--num_layers', type=int, default=130,
                help='# of channels to add to net in total')
        parser.add_argument('--dropout', type=float, default=0.2,
                help='drop out probability on classification layer')
        parser.add_argument('--cutout_prob', type=float, default=0.5,
                help='probability of using cutout')
        parser.add_argument('--min_erase_area', type=float, default=0.08,
                help='minimum size of image cutout')
        parser.add_argument('--max_erase_area', type=float, default=0.7,
                help='maximum size of image cutout')
        parser.add_argument('--min_erase_aspect_ratio', type=float, default=0.3,
                help='minimum aspect ratio of image cutout')
        parser.add_argument('--max_erase_regions', type=int, default=6,
                help='maximum number of different regions to erase')
        parser.add_argument('--shake_drop', action='store_true', default=False, 
                help='use shake drop')
        parser.add_argument('--no_se', action='store_true', default=False, 
                help='no squeeze excitation')
        parser.add_argument('--lr_finder', action='store_true', default=False, 
                help='find lr using exp lr finder')
        parser.add_argument('--wd_finder', action='store_true', default=False, 
                help='find wd using exp wd finder')
        parser.add_argument('--fold', type=int, default=0,
                help='which fold to use (0 - 4)')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = os.path.join('..', '..', 'data')
        self.path = os.path.join('..', '..', 'training_logs', self.name)
