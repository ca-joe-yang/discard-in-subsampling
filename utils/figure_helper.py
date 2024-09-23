from matplotlib import rcParams
import seaborn as sns

def set_matplotlib_sns_params(font_scale: float = 2.0) -> None:
    rcParams.update({'figure.autolayout': True})
    rcParams['mathtext.default'] = 'regular'
    sns.set_theme(
        context='paper', 
        font_scale=font_scale, 
        rc={"lines.linewidth": 3}
    )
    sns.set_palette('colorblind', 30)
    sns.set_style('ticks')
    rcParams["font.family"] = 'Times New Roman'
