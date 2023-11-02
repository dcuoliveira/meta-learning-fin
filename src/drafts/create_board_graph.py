
from collections import defaultdict
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# load board assc data
df = pd.read_csv('src/data/inputs/boardexna.csv')
print(df.columns)

# create edge list
edges = []

# get unique ticker IDs
ticker_list = df['Ticker'].dropna().unique().tolist()
ticker_list.sort()
ticker_dict = {}
cur_ind = 0
for ticker in ticker_list:
    ticker_dict[ticker] = cur_ind
    cur_ind = cur_ind + 1

# define node list
node_feats = torch.ones((cur_ind, 2), dtype=torch.float32)

# loop through common asscs
edge_added_dict = defaultdict(lambda: False)
df = df.loc[(df['StartCompanyDateStartRole'] >= '2019-01-01') & (df['StartCompanyDateStartRole'] < '2019-02-01')]
grouped = df.groupby('CompanyName')['Ticker'].unique()
for value in grouped:
    value = [str(x) for x in value]
    if 'nan' in value:
        value.remove('nan')
    val_len = len(value)
    if val_len > 1:
        try:
            value.sort()
        except:
            print(value)
            break
        for cur_src in range(0, val_len - 2):
            for cur_tgt in range(cur_src + 1, val_len - 1):
                cur_key = value[cur_src] + value[cur_tgt]
                if not edge_added_dict[cur_key]:
                    edge_added_dict[cur_key] = True
                    edges.append(torch.tensor([[ticker_dict[value[cur_src]]], [ticker_dict[value[cur_tgt]]]], dtype=torch.int64))
edges = torch.cat(edges, dim=1)

print(edges.shape)
print(node_feats.shape)

graph = Data(x=node_feats, edge_index=edges)
print(graph)

g = to_networkx(graph, to_undirected=True)
g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
print(g)

nx.draw(g, node_size=1, width=0.1)
plt.savefig("board_graph.png", dpi=1000)