from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
    'WESAD': [(0.97, 1), (0.97, 1)],
    'DREAMER': [(0.97, 1), (0.97, 1)],
    'HCI': [(0.97, 1), (0.97, 1)]
}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
    'WESAD': 0.0001,
    'DREAMER': 0.0001,
    'HCI': 0.0001,
}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
    'WESAD': (97, 1),
    'DREAMER': (97, 1),
    'HCI': (97, 1),
}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9
