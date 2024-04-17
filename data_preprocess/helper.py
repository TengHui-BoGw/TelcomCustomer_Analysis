import pandas as pd

def cut_level(data, x, interval):
    if x == 'useminutes':
        start = -1
    else:
        start = 0
    label_bins = [x for x in range(start, max(data[x]) + interval, interval)]
    label_levels = pd.cut(data[x], bins=label_bins, labels=False)
    label_levels = label_levels + 1
    return label_levels


def get_value_level(value,bins):
    if value < bins[0]:
        value_level = 1
    elif value >= bins[0] and value < bins[1]:
        value_level = 2
    elif value >= bins[1]:
        value_level = 3
    return value_level