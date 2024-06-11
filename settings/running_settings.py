import os
import seaborn as sns
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

pastel_blue_palette = ['#658EA9','#E98973', '#2B4260']
sns.palplot(sns.color_palette(pastel_blue_palette))


plots_settings ={
    'color':{
    'pastel_blue': '#658EA9'},
    'label_fontsize': 14,
    'title_fontsize': 14,
    'palette': pastel_blue_palette,
    'params':{'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 14,
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
}


run_settings={
    'blacklist':[],
    'other_parameters_and_settings': None
}