import numpy as np
from numpy import linalg as la
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import csv


N_Y_NODES = 2
# font_size = 18

def load_accoustic_data(path_data='data/EMO/', y_filename='Y.mat', x_key='dataX', y_key='dataY', all_data=False, plot_data=True, sum_data=True, figsize=(9,3), rm_extra_Y=True, standardize_Y=False):
    """
    Loads acoustic data from `X.mat` and `Y.mat`, combines them, and optionally plots and summarizes the data.

    Parameters:
    - path_data (str): Directory containing the `.mat` files. Defaults to `'data/'`.
    - y_filename (str): Name of the Y data file. Defaults to `'Y.mat'`.
    - x_key (str): Key for X data in mat file. Defaults to `'dataX'`.
    - y_key (str): Key for Y data in mat file. Defaults to `'dataY'`.
    - plot_data (bool): If True, plots the first 300 columns of the data. Defaults to True.
    - sum_data (bool): If True, prints shape, range, and mean of the data. Defaults to True.
    - figsize (tuple): Size of the plot figure. Defaults to (9, 3).
    - rm_extra_Y (bool): If True, removes extra rows from Y if it has more than 2 rows. Defaults to True.
    - standardize_Y (bool): If True, standardizes Y (mean 0, variance 1) along each row. Defaults to False.

    Returns:
    - Data (numpy.ndarray): Combined dataset from `X` and `Y`.
    """

    if all_data:
        Data = scipy.io.loadmat(path_data + 'Xall_norm.mat')['Xall'].T

        X = Data[:Data.shape[0]-2,:]
        Y = Data[-2:,:]
    else:
        X = scipy.io.loadmat(path_data + 'X.mat')[x_key].T
        Y = scipy.io.loadmat(path_data + y_filename)[y_key].T

        if rm_extra_Y and Y.shape[0] > 2:
            ### Column names from Luca's email: 
            # pleasant eventful chaotic vibrant uneventful calm annoying monotonous appropriate

            print(f'Original Y shape: {Y.shape}. Keeping only the first 2 rows.')
            Y = Y[:2, :]

        if standardize_Y:
            std_Y = Y.std(axis=1, keepdims=True)
            std_Y[std_Y == 0] = 1 # Avoid division by zero
            Y = (Y - Y.mean(axis=1, keepdims=True)) / std_Y

        Data = np.vstack((X, Y))

    N, M = Data.shape

    if sum_data:
        print(f'Shape of: X: {X.shape}  -  Y: {Y.shape}  -  data: {Data.shape}')
        print(f'(Min, Max) values of: X: ({np.min(X):.3f}, {np.max(X):.3f})  - ' + \
            f'Y: ({np.min(Y):.3f}, {np.max(Y):.3f})  -  data: ({np.min(Data):.3f}, {np.max(Data):.3f})')
        print(f'Mean value of: X: {X.mean():.3f}  -  Y: {Y.mean():.3f}  -  data: {Data.mean():.3f}')
        print(f'Std value of: X: {X.std():.3f}  -  Y: {Y.std():.3f}  -  data: {Data.std():.3f}')

    if plot_data:
        plt.figure(figsize=(figsize[0], figsize[1]*2))
        
        plt.figure()
        plt.imshow(Data[:,:300])
        plt.colorbar()
        plt.title('Data (First 300 columns)')
        
        plt.figure()
        C = np.corrcoef(Data)
        plt.imshow(C)
        plt.colorbar()
        plt.title('Correlation Matrix')
        
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
    

def save_plot_graph(A_est, th, lamb, directed=False, max_width=2, file_name=None, save=False):
    # Threshold the graph
    A_bin = np.where(np.abs(A_est) >= th, A_est, 0)
    # N = A_bin.shape[0]

    # Count proportion of edges
    edges = np.sum(np.abs(A_bin) > 0)/A_est.size

    # Create full graph
    G = nx.DiGraph(A_bin) if directed else nx.Graph(A_bin)
    node_colors = ['dodgerblue'] * (G.number_of_nodes() - N_Y_NODES) + ['red', 'red']
    edge_colors = get_edge_colors(G)

    # Get edge width
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(weights)
    normalized_widths = [(w / max_weight) * max_width for w in weights]

    # Plot
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot weighted A
    cax = axes[0].imshow(A_est)
    plt.colorbar(cax, ax=axes[0])
    axes[0].set_title(f'A, lambda={lamb:.4f}, edges={edges:.2f}')

    # Plot graph - spring layout
    pos = nx.spring_layout(G)    
    # pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=100,
            edge_color=edge_colors, linewidths=1, font_size=8, ax=axes[1],
            width=normalized_widths)


    print(f'- Lamb: {lamb}  -  Prop. edges above threshold: {edges:.3f}')

    if file_name and save:
        file_name_th = f'{file_name}'
        plt.savefig(file_name_th + '.png')
        np.save(file_name, A_est)
        scipy.io.savemat(file_name + '.mat', {'A_est': A_est, 'A_bin': A_bin})


def save_plot_subgraph(A_est, th, lamb, directed=False, node_size=900, max_width=2, 
                       font_size=18, node_font_size=16, cmap='RdBu', show_edge_weights=False,
                       file_name=None, save=False):
    # Threshold the adjacency matrix
    A_bin = np.where(np.abs(A_est) >= th, A_est, 0)

    N = A_bin.shape[0]
    edge_density = np.sum(np.abs(A_bin) > 0) / A_est.size

    G = nx.DiGraph(A_bin) if directed else nx.Graph(A_bin)

    # Extract subgraph: nodes connected to the last output nodes
    connected_to_last = subgraph_indexes(np.abs(A_bin), N)
    A_sub = A_est[np.ix_(connected_to_last, connected_to_last)]
    A_sub_bin = np.where(np.abs(A_sub) >= th, A_sub, 0)

    # Count edges connected to the output nodes
    edges = np.sum(np.abs(A_sub_bin[-2:, :]) > 0)

    # Create figure
    _, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Plot 1: Adjacency matrix with smaller colorbar ---
    vmax = np.max(np.abs(A_sub))  # Max absolute value
    im = axes[0].imshow(A_sub, cmap=cmap, vmin=-vmax, vmax=vmax)

    # Align colorbar to the right, same height
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    axes[0].set_xticks(range(len(connected_to_last)))
    axes[0].set_yticks(range(len(connected_to_last)))
    # axes[0].set_xticklabels([i + 1 for i in connected_to_last], fontsize=font_size)
    axes[0].set_xticklabels([i + 1 for i in connected_to_last], fontsize=font_size, rotation=45)

    axes[0].set_yticklabels([i + 1 for i in connected_to_last], fontsize=font_size)

    axes[0].set_xlabel("Node index", fontsize=font_size)
    axes[0].set_ylabel("Node index", fontsize=font_size)

    # --- Plot 2: Subgraph ---
    G_sub = nx.DiGraph(A_sub_bin) if directed else nx.Graph(A_sub_bin)
    node_colors_sub = ['lightskyblue'] * (G_sub.number_of_nodes() - N_Y_NODES) + ['lightcoral', 'lightcoral']
    edge_colors_sub = get_edge_colors(G_sub)

    # Compute edge widths
    weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
    max_weight = max(weights) if weights else 1  # Avoid division by zero
    normalized_widths = [(w / max_weight) * max_width for w in weights]

    # Node labels with 1-based indexing
    labels_sub = {node: label + 1 for node, label in zip(G_sub.nodes, connected_to_last)}
    pos_sub = nx.circular_layout(G_sub)

    nx.draw(
        G_sub, pos_sub, with_labels=True, labels=labels_sub, node_color=node_colors_sub,
        node_size=node_size, edge_color=edge_colors_sub, linewidths=2.5, font_size=node_font_size, ax=axes[1],
        width=normalized_widths
    )

    if show_edge_weights:
        edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G_sub.edges(data=True)}
        nx.draw_networkx_edge_labels(G_sub, pos_sub, edge_labels=edge_labels, font_size=12, ax=axes[1])

    print(f'- Lambda: {lamb:.4f}  -  Edge density above threshold: {edge_density:.3f}  -  Output edges: {edges}')
    
    if file_name and save:
        fig_name_th = f'{file_name}_sub'
        plt.savefig(fig_name_th + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(fig_name_th + '.pdf', bbox_inches='tight')
        np.save(file_name, A_est)
        scipy.io.savemat(file_name + '.mat', {'A_est': A_est, 'A_sub': A_sub, 'A_sub_bin': A_sub_bin})
        print('\tSaved as:', file_name)


def save_plot_subgraph_grid(A_est, thresholds, lamb, directed=False, 
                            font_size=16, cmap='RdBu', 
                            file_name=None, save=False, th_fixed_subgraph=None,
                            plot_graph=False, title=False):
    
    # If saving, prepare directory
    if save and file_name:
        import os
        base_dir = os.path.dirname(file_name)
        if not base_dir:
            base_dir = '.'
        save_dir = os.path.join(base_dir, 'thresholds')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    num_plots = len(thresholds)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    # Ensure axes is iterable even if num_plots is 1
    if num_plots == 1:
        axes = [axes]

    if plot_graph:
        fig_graph, axes_graph = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1:
            axes_graph = [axes_graph]
        
    N = A_est.shape[0]

    # If th_fixed_subgraph is provided, compute the fixed subgraph nodes
    if th_fixed_subgraph is not None:
        A_bin_fixed = np.where(np.abs(A_est) >= th_fixed_subgraph, A_est, 0)
        connected_to_last_fixed = subgraph_indexes(np.abs(A_bin_fixed), N)

    for i, th in enumerate(thresholds):
        ax = axes[i]
        
        # Threshold the adjacency matrix
        A_bin = np.where(np.abs(A_est) >= th, A_est, 0)
        
        # Extract subgraph: nodes connected to the last output nodes
        # Use fixed subgraph nodes if th_fixed_subgraph is provided, else compute dynamically
        if th_fixed_subgraph is not None:
            connected_to_last = connected_to_last_fixed
        else:
            connected_to_last = subgraph_indexes(np.abs(A_bin), N)
        
        if len(connected_to_last) == 0:
            ax.text(0.5, 0.5, "No connected nodes", ha='center', va='center')
            if title:
                ax.set_title(f'Th={th}')
            if plot_graph:
                axes_graph[i].text(0.5, 0.5, "No connected nodes", ha='center', va='center')
                if title:
                    axes_graph[i].set_title(f'Th={th}')
            continue
            
        A_sub = A_est[np.ix_(connected_to_last, connected_to_last)]
        
        # Apply current threshold to the subgraph for display purposes
        A_sub_display = np.where(np.abs(A_sub) >= th, A_sub, 0)
        
        vmax = np.max(np.abs(A_sub_display)) if A_sub_display.size > 0 else 1
        im = ax.imshow(A_sub_display, cmap=cmap, vmin=-vmax, vmax=vmax)
        
        if title:
            ax.set_title(f'Th={th}')
        
        # Ticks and labels
        ax.set_xticks(range(len(connected_to_last)))
        ax.set_yticks(range(len(connected_to_last)))
        ax.set_xticklabels([idx + 1 for idx in connected_to_last], fontsize=font_size-4, rotation=90)
        ax.set_yticklabels([idx + 1 for idx in connected_to_last], fontsize=font_size-4)
        
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=font_size-4)

        # --- Individual Save Logic ---
        if save and file_name:
            # Re-create individual plot to save cleanly
            fig_ind, ax_ind = plt.subplots(figsize=(6, 6))
            im_ind = ax_ind.imshow(A_sub_display, cmap=cmap, vmin=-vmax, vmax=vmax)
            if title:
                ax_ind.set_title(f'Th={th}')
            ax_ind.set_xticks(range(len(connected_to_last)))
            ax_ind.set_yticks(range(len(connected_to_last)))
            ax_ind.set_xticklabels([idx + 1 for idx in connected_to_last], fontsize=font_size-4, rotation=90)
            ax_ind.set_yticklabels([idx + 1 for idx in connected_to_last], fontsize=font_size-4)
            
            divider_ind = make_axes_locatable(ax_ind)
            cax_ind = divider_ind.append_axes("right", size="5%", pad=0.15)
            cbar_ind = plt.colorbar(im_ind, cax=cax_ind)
            cbar_ind.ax.tick_params(labelsize=font_size-4)
            
            plt.tight_layout()
            
            # Construct filename
            base_name = os.path.basename(file_name)
            save_path = os.path.join(save_dir, f'{base_name}_th_{th}.png')
            fig_ind.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig_ind)

        # Plot Graph Topology if requested
        if plot_graph:
            ax_g = axes_graph[i]
            G_sub = nx.DiGraph(A_sub_display) if directed else nx.Graph(A_sub_display)
            
            # Node colors (last 2 are outputs)
            node_colors_sub = ['lightskyblue'] * (len(connected_to_last) - N_Y_NODES) 
            # Handle case where fewer than N_Y_NODES exist (though unlikely given subgraph logic)
            if len(connected_to_last) >= N_Y_NODES:
                node_colors_sub.extend(['lightcoral'] * N_Y_NODES)
            else:
                 node_colors_sub = ['lightcoral'] * len(connected_to_last)

            edge_colors_sub = get_edge_colors(G_sub)

            # Compute edge widths
            weights = [G_sub[u][v]['weight'] for u, v in G_sub.edges()]
            max_weight = max(weights) if weights else 1
            max_width = 2
            normalized_widths = [(w / max_weight) * max_width for w in weights]

            # Layout
            pos_sub = nx.circular_layout(G_sub)
            
            # Labels
            labels_sub = {node: label + 1 for node, label in zip(G_sub.nodes, connected_to_last)}

            nx.draw(
                G_sub, pos_sub, with_labels=True, labels=labels_sub, node_color=node_colors_sub,
                node_size=500, edge_color=edge_colors_sub, linewidths=1.5, font_size=font_size-4, ax=ax_g,
                width=normalized_widths
            )
            if title:
                ax_g.set_title(f'Graph Th={th}')

            # --- Individual Graph Save Logic ---
            if save and file_name:
                fig_g_ind, ax_g_ind = plt.subplots(figsize=(6, 6))
                nx.draw(
                    G_sub, pos_sub, with_labels=True, labels=labels_sub, node_color=node_colors_sub,
                    node_size=500, edge_color=edge_colors_sub, linewidths=1.5, font_size=font_size-4, ax=ax_g_ind,
                    width=normalized_widths
                )
                if title:
                    ax_g_ind.set_title(f'Graph Th={th}')
                
                base_name = os.path.basename(file_name)
                save_path_g = os.path.join(save_dir, f'{base_name}_graph_th_{th}.png')
                fig_g_ind.savefig(save_path_g, dpi=300, bbox_inches='tight')
                plt.close(fig_g_ind)

    plt.tight_layout()
    if plot_graph:
        fig_graph.tight_layout()
    
    if save and file_name and plot_graph:
        print(f'\tIndividual plots saved in: {save_dir}')


def compute_err_sparsity(A_ests, X, th, th_err=False, target_idx=None):
    sparsity = np.zeros(len(A_ests))
    err = np.zeros(len(A_ests))

    X_target = X if target_idx is None else X[target_idx, :]
    X_norm = la.norm(X_target)

    for i, A_est in enumerate(A_ests):

        A_est = A_est if target_idx is None else A_est[target_idx, :]
        A_th = np.where(np.abs(A_est) >= th, A_est, 0)

        if th_err:
            err[i] = (la.norm( A_th@X - X_target ) / X_norm )**2
        else:
            err[i] = (la.norm( A_est@X - X_target ) / X_norm )**2

        n_links = np.sum(np.abs(A_est) > th)
        sparsity[i] = n_links / 2 if target_idx is None else n_links

    return np.array(err), np.array(sparsity)


def save_plot_err_sparsity(lambdas, err, sparsity, title='Error - all X', file_name=None, save=False, logx=False):
    # PLOT
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Primera gráfica: err vs Mults
    if logx:
        axes[0].semilogx(lambdas, err, marker='o', linestyle='-', color='b')
    else:
        axes[0].plot(lambdas, err, marker='o', linestyle='-', color='b')
    axes[0].set_xlabel('Lambda')
    axes[0].set_ylabel('Error')
    axes[0].set_title(title)
    axes[0].grid(True)

    # Segunda gráfica: err vs sparsity
    axes[1].plot(sparsity, err, marker='o', linestyle='-', color='b')
    axes[1].set_xlabel('Number of links')
    axes[1].set_ylabel('Error')
    axes[1].set_title(title)
    axes[1].grid(True)

    # Mostrar la figura
    plt.tight_layout()  # Ajusta los subgráficos para que no se superpongan
    plt.show()

    if file_name and save:
        fig.savefig(file_name + '.png')
        np.savez(file_name + '.npz', lambdas=lambdas, sparsity=sparsity, err=err)
        df = pd.DataFrame({'lambdas': lambdas, 'sparsity': sparsity, 'err': err})
        df.to_csv(file_name + '.csv', index=False)
        print('\t Saved as:', file_name)


def sort_edges_by_weight(A_est, th, output_file=None, abs_val=True, save=False):
    # Apply threshold
    A_bin = np.where(np.abs(A_est) >= th, A_est, 0)

    # Subgraph: nodes connected to output
    N = A_bin.shape[0]
    A_sub = A_est[-2:, :]
        
    edges = []
    N = A_est.shape[1]
    for i in range(2):  # 0 and 1 = last two rows
        source_node = A_est.shape[0] - 2 + i
        for j in range(N):
            weight = A_sub[i, j]
            if abs(weight) >= th and source_node != j:
                edges.append((source_node + 1, j + 1, weight))

    # Sort by weight (absolute if specified)
    if abs_val:
        edges.sort(key=lambda x: abs(x[2]), reverse=True)
    else:
        edges.sort(key=lambda x: x[2], reverse=True)

    if save:
        edges_np_file = output_file + '_edges.csv'
        with open(edges_np_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "weight"])  # header
            for edge in edges:
                writer.writerow(edge)

    text = ""
    for i, j, w in edges:
        text += f"({i} → {j}): weight = {w:.4f}\n"

    if output_file is None:
        print(text)
    else:
        output_file += '_edges.txt'
        with open(output_file, 'w') as f:
            f.write(text)

    return edges
