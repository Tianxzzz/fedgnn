

import os
import sys

import pickle as pkl
import numpy as np


def generate_partial_graphs(dataname):
    if dataname == 'METR-LA':
        datadir = 'data/{}'.format(dataname)
        adj_mx_f = 'data/sensor_graph/adj_mx.pkl'
    else:
        datadir = 'data/{}'.format(dataname)
        adj_mx_f = 'data/sensor_graph/adj_mx_bay.pkl'
    with open(adj_mx_f, 'rb') as f:
        adj_mx = pkl.load(f, encoding='latin1')
    print(adj_mx)

    if dataname == 'METR-LA':
        sensor_locations = np.genfromtxt('data/sensor_graph/graph_sensor_locations.csv', delimiter=',',
                                         skip_header=1)
        for t in range(sensor_locations.shape[0]):
            assert int(sensor_locations[t, 1]) == int(adj_mx[0][t])
    else:
        sensor_locations = np.genfromtxt('data/sensor_graph/graph_sensor_locations_bay.csv',
                                         delimiter=',', skip_header=0)
        for t in range(sensor_locations.shape[0]):
            assert int(sensor_locations[t, 0]) == int(adj_mx[0][t])

    # sort all sensors via longitude
    long_sorted_sensors = np.argsort(sensor_locations[:, -1])

    resdict = {}
    for ratio in [0.05, 0.25, 0.5, 0.75, 0.9, 1.0]:
        num_nodes = np.round(len(long_sorted_sensors) * ratio).astype(int)
        selected_nodes = sorted(long_sorted_sensors[:num_nodes])
        unselected_nodes = sorted(long_sorted_sensors[num_nodes:])
        print(selected_nodes, unselected_nodes)
        selected_edges = adj_mx[2][selected_nodes, :][:, selected_nodes]


        x, y = sensor_locations[selected_nodes][:, -2], sensor_locations[selected_nodes][:, -1]

        if ratio < 1:
            resdict[str(ratio)] = (selected_nodes, unselected_nodes)
    np.savez('data/sensor_graph/{}_partial_nodes.npz'.format(dataname), **resdict)


generate_partial_graphs('METR-LA')
