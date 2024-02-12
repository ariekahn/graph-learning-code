import networkx as nx
import pandas as pd

_modular_list = {
    0: [1, 2, 3, 14],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [1, 2, 3, 5],
    5: [4, 6, 7, 8],
    6: [5, 7, 8, 9],
    7: [5, 6, 8, 9],
    8: [5, 6, 7, 9],
    9: [6, 7, 8, 10],
    10: [9, 11, 12, 13],
    11: [10, 12, 13, 14],
    12: [10, 11, 13, 14],
    13: [10, 11, 12, 14],
    14: [11, 12, 13, 0],
}

_ring_lattice_list = {
    0: [13, 14, 1, 2],
    1: [14, 0, 2, 3],
    2: [0, 1, 3, 4],
    3: [1, 2, 4, 5],
    4: [2, 3, 5, 6],
    5: [3, 4, 6, 7],
    6: [4, 5, 7, 8],
    7: [5, 6, 8, 9],
    8: [6, 7, 9, 10],
    9: [7, 8, 10, 11],
    10: [8, 9, 11, 12],
    11: [9, 10, 12, 13],
    12: [10, 11, 13, 14],
    13: [11, 12, 14, 0],
    14: [12, 13, 0, 1],
}

modular = nx.from_dict_of_lists(_modular_list)
ring_lattice = nx.from_dict_of_lists(_ring_lattice_list)

subjects = ['GLS003', 'GLS004', 'GLS005', 'GLS006',
            'GLS008', 'GLS009', 'GLS010', 'GLS011',
            'GLS013', 'GLS014', 'GLS017', 'GLS018',
            'GLS019', 'GLS020', 'GLS021', 'GLS022',
            'GLS023', 'GLS024', 'GLS025', 'GLS026',
            'GLS027', 'GLS028', 'GLS030', 'GLS033',
            'GLS037', 'GLS038', 'GLS039', 'GLS040',
            'GLS043', 'GLS044', 'GLS045']

graph = (('GLS003', 'Modular'), ('GLS004', 'Modular'), ('GLS005', 'Lattice'), ('GLS006', 'Lattice'),
         ('GLS008', 'Modular'), ('GLS009', 'Lattice'), ('GLS010', 'Modular'), ('GLS011', 'Modular'),
         ('GLS013', 'Modular'), ('GLS014', 'Lattice'), ('GLS016', 'Modular'), ('GLS017', 'Lattice'),
         ('GLS018', 'Modular'), ('GLS019', 'Modular'), ('GLS020', 'Lattice'), ('GLS021', 'Lattice'),
         ('GLS022', 'Modular'), ('GLS023', 'Modular'), ('GLS024', 'Lattice'), ('GLS025', 'Modular'),
         ('GLS026', 'Modular'), ('GLS027', 'Lattice'), ('GLS028', 'Lattice'), ('GLS030', 'Lattice'),
         ('GLS033', 'Lattice'), ('GLS034', 'Lattice'), ('GLS036', 'Modular'), ('GLS037', 'Modular'),
         ('GLS038', 'Modular'), ('GLS039', 'Modular'), ('GLS040', 'Lattice'), ('GLS041', 'Lattice'),
         ('GLS043', 'Lattice'), ('GLS044', 'Modular'), ('GLS045', 'Lattice'),
         )
graph_df = pd.DataFrame(graph, columns=('subject', 'graph'))
graph_df = graph_df.loc[graph_df.subject.isin(subjects)].reset_index(drop=True)

subjects_modular = list(graph_df[graph_df['graph'] == 'Modular'].subject.values)
subjects_lattice = list(graph_df[graph_df['graph'] == 'Lattice'].subject.values)
