import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100', magenta='#e87ba4', green='#008300', violet='#4a3aa7', red='#e34948', ink='#0b0b0b', mute='#52514e', grid='#d9d8d3')
LES = dict(MA='#1baf7a', HE='#e34948', EX='#eda100', SE='#2a78d6')       # lesion overlay colours (consistent across all figures)
GRADES = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR', 'Ungradable']
mpl.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': C['mute'], 'axes.linewidth': .6,
                     'axes.grid': True, 'grid.color': C['grid'], 'grid.linewidth': .5, 'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'axes.axisbelow': True})
OUT = '/home/user/Diabetic-Retinopathy/paper/figs/'
