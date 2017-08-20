import matplotlib.pyplot as plt
import numpy as np
from math import log10

def colSel(t):
    if t == 'loss':
        return 'red'
    elif t == 'acc':
        return 'c'
    elif t == 'val_loss':
        return 'tomato'
    elif t == 'val_acc':
        return 'skyblue'
    elif t == 'lr':
        return 'silver'


def plotCurve(l_str, l_val, lr_init, r_str=None, r_val=None):
    e = len(l_val)
    x = np.linspace(1, e, e, dtype=int)
    text_x = max(x)*0.2
    text_y = max(l_val)*0.8
    fig=plt.figure()
    ax_l = fig.add_subplot(111)

    ax_l.plot(x, l_val, color=colSel(l_str))
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel(l_str)
    ax_l.yaxis.label.set_color(colSel(l_str))
    ax_l.text(text_x, text_y, r'$Initial Learning Rate = %f$' % lr_init)

    if l_str == 'lr':
        raise Exception('Only r_str accepts \'lr\'.')

    if r_str is not None and r_val is not None:
        ax_r = ax_l.twinx()
        if r_str == 'lr':
            r_val_log10 = [log10(x / r_val[0]) for x in r_val]
            r_val = r_val_log10
            ax_r.set_ylim([-5.0, 0.05])

        ax_r.plot(x, r_val, color=colSel(r_str))
        ax_r.yaxis.label.set_color(colSel(r_str))
        ax_r.set_ylabel(r_str)


    plt.show()


if __name__ == '__main__':
    l_v = [1.400, 0.800, 0.600, 0.400, 0.300, 0.250, 0.200, 0.170, 0.140, 0.115]
    a_v = [0.400, 0.600, 0.800, 0.900, 0.930, 0.950, 0.960, 0.970, 0.975, 0.980]
    r_v = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.00001, 0.00001, 0.00001, 0.000001, 0.000001]
    plotCurve(l_str='loss',l_val=l_v, r_str='lr', r_val=r_v)
