import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx


N_Y_NODES = 2


def load_accoustic_data(path_data='data/', plot_data=True, sum_data=True, figsize=(9,3)):
    """
    Loads acoustic data from `X.mat` and `Y.mat`, combines them, and optionally plots and summarizes the data.

    Parameters:
    - path_data (str): Directory containing the `.mat` files. Defaults to `'data/'`.
    - plot_data (bool): If True, plots the first 300 columns of the data. Defaults to True.
    - sum_data (bool): If True, prints shape, range, and mean of the data. Defaults to True.
    - figsize (tuple): Size of the plot figure. Defaults to (9, 3).

    Returns:
    - Data (numpy.ndarray): Combined dataset from `X` and `Y`.
    """

    X = scipy.io.loadmat(path_data + 'X.mat')['dataX'].T
    Y = scipy.io.loadmat(path_data + 'Y.mat')['dataY'].T
    Data = np.vstack((X, Y))

    N, M = Data.shape

    if sum_data:
        print(f'Shape of: X: {X.shape}  -  Y: {Y.shape}  -  data: {Data.shape}')
        print(f'(Min, Max) values of: X: ({np.min(X):.3f}, {np.max(X):.3f})  - ' + \
            f'Y: ({np.min(Y):.3f}, {np.max(Y):.3f})  -  data: ({np.min(Data):.3f}, {np.max(Data):.3f})')
        print(f'Mean value of: X: {X.mean():.3f}  -  Y: {Y.mean():.3f}  -  data: {Data.mean():.3f}')

    if plot_data:
        plt.figure(figsize=figsize)
        plt.imshow(Data[:,:300])
        plt.colorbar()
        plt.title('Data')
        plt.tight_layout()

    return Data


def get_edge_colors(G):
    """
    Assigns colors to the edges of a graph based on their connection to specific nodes.

    Parameters:
    - G (networkx.Graph): The input graph.

    Returns:
    - edge_colors (list): A list of colors ('lime' or 'grey') for each edge in the graph.
      Edges connected to the last `N_Y_NODES` nodes are colored 'lime'; others are 'grey'.
    """

    edge_colors = []
    N = G.number_of_nodes()
    for u, v in G.edges():
        if u >= N-N_Y_NODES or v >= N-N_Y_NODES:
            edge_colors.append('lime')
        else:
            edge_colors.append('grey')
    return edge_colors


def subgraph_indexes(A_bin, N):
    connected_to_last = set([N-2, N-1])
    connected_to_last.update(np.where(A_bin[-2] > 0)[0])
    connected_to_last.update(np.where(A_bin[:,-2] > 0)[0])
    connected_to_last.update(np.where(A_bin[-1] > 0)[0])
    connected_to_last.update(np.where(A_bin[:,-1] > 0)[0])
    connected_to_last = sorted(connected_to_last)
    return connected_to_last
    

def save_plot_graph(A_est, th, lamb, directed=False, file_name=None, save=False):
    # Threshold the graph
    A_bin = np.where(A_est >= th, 1, 0)
    N = A_bin.shape[0]

    # Count proportion of edges
    edges = np.sum(A_bin)/A_est.size

    # Create full graph
    G = nx.DiGraph(A_bin) if directed else nx.Graph(A_bin)
    node_colors = ['dodgerblue'] * (G.number_of_nodes() - N_Y_NODES) + ['red', 'red']
    edge_colors = get_edge_colors(G)

    # Create subgraph
    connected_to_last = subgraph_indexes(A_bin, N)
    A_sub = A_est[np.ix_(connected_to_last, connected_to_last)]
    A_sub_bin = np.where(A_sub >= th, 1, 0)

    # Plot
    _, axes = plt.subplots(1, 4, figsize=(32, 8))

    # Plot weighted A
    cax = axes[0].imshow(A_est)
    plt.colorbar(cax, ax=axes[0])
    axes[0].set_title(f'A, lambda={lamb:.4f}, edges={edges:.2f}')

    # Plot graph - spring layout
    pos = nx.spring_layout(G)    
    # pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=100,
            edge_color=edge_colors, linewidths=1, font_size=8, ax=axes[1])

    # Plot weighted A - subgraph
    cax = axes[2].imshow(A_sub)
    plt.colorbar(cax, ax=axes[2])
    axes[2].set_xticks(range(len(connected_to_last)))
    axes[2].set_yticks(range(len(connected_to_last)))
    axes[2].set_xticklabels(connected_to_last)
    axes[2].set_yticklabels(connected_to_last)
    axes[2].set_title(f'A sub, lambda={lamb:.4f}, edges={edges:.2f}')

    G_sub = nx.DiGraph(A_sub_bin) if directed else nx.Graph(A_sub_bin)
    node_colors_sub = ['dodgerblue'] * (G_sub.number_of_nodes() - N_Y_NODES) + ['red', 'red']
    edge_colors_sub = get_edge_colors(G_sub)

    labels_sub = {node: label for node, label in zip(G.nodes, connected_to_last)}
    pos_sub = nx.circular_layout(G_sub)
    nx.draw(G_sub, pos_sub, with_labels=True, labels=labels_sub, node_color=node_colors_sub, 
            node_size=200, edge_color=edge_colors_sub, linewidths=2.5, font_size=12, ax=axes[3])

    print(f'- Lamb: {lamb}  -  Prop. edges above threshold: {edges:.3f}')

    if file_name and save:
        file_name_th = f'{file_name}_{th}'
        plt.savefig(file_name_th + '.png')
        np.save(file_name + '.npz', A_est)
        scipy.io.savemat(file_name + '.mat', {'A_est': A_est, 'A_bin': A_bin})


def save_plot_subgraph(A_est, th, lamb, directed=False, file_name=None, save=False):
    # Threshold the graph
    A_bin = np.where(A_est >= th, 1, 0)
    N = A_bin.shape[0]
    edges = np.sum(A_bin)/A_est.size
    G = nx.DiGraph(A_bin) if directed else nx.Graph(A_bin)

    # Create subgraph
    connected_to_last = subgraph_indexes(A_bin, N)
    A_sub = A_est[np.ix_(connected_to_last, connected_to_last)]
    A_sub_bin = np.where(A_sub >= th, 1, 0)

    # Plot
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot weighted A - subgraph
    cax = axes[0].imshow(A_sub)
    plt.colorbar(cax, ax=axes[0])
    axes[0].set_xticks(range(len(connected_to_last)))
    axes[0].set_yticks(range(len(connected_to_last)))
    axes[0].set_xticklabels(connected_to_last)
    axes[0].set_yticklabels(connected_to_last)
    axes[0].set_title(f'A sub, lambda={lamb:.4f}, edges={edges:.2f}')

    G_sub = nx.DiGraph(A_sub_bin) if directed else nx.Graph(A_sub_bin)
    node_colors_sub = ['dodgerblue'] * (G_sub.number_of_nodes() - N_Y_NODES) + ['red', 'red']
    edge_colors_sub = get_edge_colors(G_sub)

    labels_sub = {node: label for node, label in zip(G.nodes, connected_to_last)}
    pos_sub = nx.circular_layout(G_sub)
    nx.draw(G_sub, pos_sub, with_labels=True, labels=labels_sub, node_color=node_colors_sub, 
            node_size=200, edge_color=edge_colors_sub, linewidths=2.5, font_size=12, ax=axes[1])

    print(f'- Lamb: {lamb}  -  Prop. edges above threshold: {edges:.3f}')

    if file_name and save:
        fig_name_th = f'{file_name}_sub_{th}'
        plt.savefig(fig_name_th + '.png')
        np.save(file_name + '.npz', A_est)
        scipy.io.savemat(file_name + '.mat', {'A_est': A_est, 'A_sub': A_sub, 'A_sub_bin': A_sub_bin})