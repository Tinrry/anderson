from utils import plot_loss_from_h5, plot_loss_scale
from utils import load_config

# config = load_config('configs/config_L6_7.json')
# plot_loss_scale(config['config_loss'])

config = load_config('config_debug.json')
plot_loss_scale(config['config_loss'])
