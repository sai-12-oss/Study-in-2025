import numpy as np
import datetime as dt


def get_times(data):
    time = [dt.datetime(*data[i][:5]) for i in range(len(data))]
    times = []

    for i in range(len(data)):
        delta = time[i] - time[0]
        times.append(delta.days + delta.seconds / 86400)

    return np.array(times)


def get_oppositions(data):
    oppositions = [data[i][5] * 30 + data[i][6] + data[i][7] / 60 + data[i][8] / 3600 for i in range(len(data))]
    return np.array(oppositions)
