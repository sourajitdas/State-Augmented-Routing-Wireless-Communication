import random
import networkx as nx
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from collections import defaultdict
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.animation import FuncAnimation
import os


# Create the source destination pair for a single flow
def source_destination(N):
    # Choose the source and destination for a given flow
    src = torch.randint(0, N, (1,), device=device_arg)
    dest = torch.randint(0, N, (1,), device=device_arg)

    while dest == src:
        dest = torch.randint(0, N, (1,), device=device_arg)

    path_tensor = torch.zeros(N)
    path_tensor[src] = -1
    path_tensor[dest] = 1

    return path_tensor


# Create the map where the source destination pairs are generated for each flow in the network
def flow_path(N, K):

    temp_path = torch.zeros(K, N)

    for k in range(K):
        final = source_destination(N)

        # Check if this path has already been chosen
        while torch.sum(torch.all(temp_path[:k, :] == final, dim=1)):
            final = source_destination(N)

        temp_path[k, :] = final

    ultimate_path = torch.transpose(temp_path, dim0=0, dim1=1)

    return ultimate_path


# Generate the probability matrix R i.e. R_ij
def generate_prob_matrix(A, dc):
    # Calculate probability values for non-zero values of A
    P = np.where(A > 0, 1 - A / dc, 1e-10)

    # Ensure non-negative probability
    P = np.where(P < 0, 1e-10, P)

    np.fill_diagonal(P, 0)

    return P


# Create the PyG graph from the networkx graph
def create_pyg_graph(norm_matrix):
    edge_indices = torch.nonzero(torch.tensor(norm_matrix), as_tuple=False).t().contiguous()

    edge_weights = torch.tensor(norm_matrix)[edge_indices[0], edge_indices[1]]

    return edge_indices, edge_weights


# Create the networkx graph
def create_network(N, dc):
    k = 4  # k-Nearest Neighbors

    # Generate some random data
    points = 1 * np.random.rand(N, 2)  # 2D coordinates

    # Create a graph
    G = nx.Graph()

    # Add nodes with their positions as attributes
    for i in range(N):
        G.add_node(i, pos=tuple(points[i]))

    # Compute pairwise distances between nodes
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            distances[i, j] = distances[j, i] = np.linalg.norm(points[i] - points[j])

    # Connect each node to its k nearest neighbors within dc
    for i in range(N):
        # Sort nodes by distance to the current node
        sorted_nodes = np.argsort(distances[i])
        connected_neighbors = 0  # To keep track of how many neighbors we've connected to
        for j in sorted_nodes:
            if j != i:  # Skip the current node to prevent self-loops
                if distances[i, j] <= dc:
                    G.add_edge(i, j, weight=distances[i, j])
                    connected_neighbors += 1
                if connected_neighbors == k:
                    break

    A = nx.to_numpy_array(G, weight='weight')
    np.fill_diagonal(A, 0)

    return A, G


def visualize_network(N, dc, G):
    # Visualize the graph
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True)

    # Drawing circles around nodes
    for node, coordinates in pos.items():
        circle = plt.Circle(coordinates, dc, fill=False, edgecolor='r', linestyle='--')
        ax.add_patch(circle)

    plt.title("Geometric graph for N=%d nodes" % N)
    plt.show()


# Create the dataset to be fed as input to the GNN model
def create_graph_data(rand_seed, samples, N, K, T, batch_size, dc):

    print("Generating new datasets for random seed", rand_seed)
    data_list = defaultdict(list)
    loader = {}

    for phase in samples:
        dbar = tqdm(range(samples[phase]), desc=f"Generating samples for {phase} = {samples[phase]} samples")

        for _ in dbar:
            distance_matrix, network_graph = create_network(N, dc)
            # Make sure to get a connected graph so we every node is connected to every other node in the network
            while not nx.is_connected(network_graph):
                distance_matrix, network_graph = create_network(N, dc)

            # visualize_network(N, dc, network_graph)

            all_path = []
            all_edge_indices = []
            all_edge_weights = []
            all_eigenvalues = []
            all_netgraph = []
            all_pos = []

            pos_dict = nx.get_node_attributes(network_graph, 'pos')
            positions = torch.tensor(list(pos_dict.values()), dtype=torch.float)

            prob_matrix = generate_prob_matrix(distance_matrix, dc)
            eigenvalues, _ = np.linalg.eig(prob_matrix)
            eigen_max = np.max(eigenvalues)
            norm_prob_matrix = prob_matrix / eigen_max

            path = flow_path(N, K)

            # Initialize tensors
            flow_num_tensor = torch.zeros((N, K, T), dtype=torch.int64)
            source_node_tensor = torch.zeros((N, K, T), dtype=torch.int64)
            packet_num_tensor = torch.zeros((N, K, T), dtype=torch.int64)
            time_step_tensor = torch.zeros((N, K, T), dtype=torch.int64)

            for t in range(T):
                # A0_t = 1 * torch.zeros(N,K)

                # Iterate through each flow to initialize packets at the source
                for k in range(K):
                    # Reset the packet ID counter for each flow
                    global_packet_id_counter = 1
                    source_node = torch.where(path[:, k] == -1)[0]  # Find the source node for flow k

                    # Randomly determine the number of packets to generate for this source and time step
                    num_packets_to_generate = torch.randint(1, 6, (1,), device=device_arg) # You can adjust the range as needed

                    for _ in range(num_packets_to_generate):
                        # Assign a unique packet ID as a combination of flow number, source node number,
                        # packet number, and time step number
                        flow_num_tensor[source_node.item(), k, t] = k
                        source_node_tensor[source_node.item(), k, t] = source_node.item()
                        packet_num_tensor[source_node.item(), k, t] = global_packet_id_counter
                        time_step_tensor[source_node.item(), k, t] = t

                        global_packet_id_counter += 1  # Increment the packet ID counter


                all_path.append(path)

                adj_matrix = nx.to_numpy_array(network_graph)
                all_netgraph.append(torch.from_numpy(adj_matrix).float())
                all_pos.append(positions)

                edge_index_t, edge_weights_t = create_pyg_graph(norm_prob_matrix)

                all_edge_indices.append(edge_index_t)
                all_edge_weights.append(edge_weights_t)
                all_eigenvalues.append(np.max(eigenvalues))

            # X = torch.stack(all_A0s, dim=2)
            path_final = torch.stack(all_path, dim=2)
            netgraph_final = torch.stack(all_netgraph, dim=2)
            pos_final = torch.stack(all_pos, dim=2)

            # Encapsulate the input packet tensors into a structured tensor
            A0_t_structured = {
                'flow_num': flow_num_tensor,
                'source_node': source_node_tensor,
                'packet_num': packet_num_tensor,
                'time_step': time_step_tensor
            }

            GSO = Data(x=A0_t_structured, edge_index=all_edge_indices, edge_attr=all_edge_weights, eig=all_eigenvalues,
                       map=path_final, network=netgraph_final, pos=pos_final)

            data_list[phase].append(GSO)

        loader[phase] = DataLoader(data_list[phase], batch_size=batch_size, shuffle=(phase == 'train'))

    return loader


# Create the GNN model class. Start with the GNN backbone.
class gnn_basic(nn.Module):
    def __init__(self, num_features_list):

        super(gnn_basic, self).__init__()
        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            # self.layers.append(TAGConv(num_features_list[i], num_features_list[i+1], K=5))
            self.layers.append(TAGConv(num_features_list[i], num_features_list[i + 1]))

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        return x


def normalize_features(input):
    # Compute mean and std along the feature dimension
    mean = input.mean(dim=0, keepdim=True)
    std = input.std(dim=0, keepdim=True)
    std[std == 0] = 1  # Prevent division by zero

    return (input - mean) / std, mean, std



# Now create the GNN model function forward pass.
class GNN_Model(nn.Module):
    def __init__(self, num_feature_list):
        super(GNN_Model, self).__init__()
        self.gnn_basic = gnn_basic(num_feature_list)
        self.b_p = nn.Linear(num_feature_list[-1], 1, bias=False)   # for the a_i^k matrix
        self.c_p = nn.Linear(num_feature_list[-1], 1, bias=False)   # for the T_i^k matrix
        self.w = nn.Linear(num_feature_list[-1], num_feature_list[-1], bias=False)  # for the square matrix

    def forward(self, x, edge_index, edge_weight, K, batch_size, N, eigen, a_0, device):

        temp_edge_weight = edge_weight.view(K, batch_size*N*(N-1))
        temp_edge_weight = temp_edge_weight.view(K, batch_size, N*(N-1))
        temp_eigen = eigen.view(K, batch_size)

        R = torch.zeros(K, batch_size, N, N, device=device)

        for k in range(K):
            for b in range(batch_size):
                idx = edge_index[:,:N*(N-1)]
                wt = temp_edge_weight[k, b, :]
                batch = torch.zeros(N, dtype=torch.long, device=device)
                adj_matrix = to_dense_adj(idx, batch, wt, max_num_nodes=N)
                R1 = torch.squeeze(adj_matrix) * temp_eigen[k, b]
                R[k,b,:,:] = R1

        y = self.gnn_basic(x, edge_index, edge_weight)  #### KB x m x F

        a_y = self.b_p(y)
        t_y = self.c_p(y)

        yw = self.w(y)


        yw = yw.view(K * batch_size, N, -1)
        y = y.view(K * batch_size, N, -1)


        K_y = torch.bmm(yw, torch.transpose(y, dim0=2, dim1=1))

        a0, mu_, path = torch.split(x, 1, dim=-1)

        path_mod = torch.clamp(path, min=0)
        path_mod = path_mod.view(K * batch_size * N, -1)


        a_y = a_y.view(K, batch_size, N)


        ######## Third Constraint Implicit Satisfaction  ########
        a_ik = a_0 + torch.relu(a_y).view(K * batch_size * N, -1)


        a_ik = torch.squeeze(a_ik)
        a_ik = a_ik.view(K, -1)

        path_mod = path_mod.view(K*batch_size, -1)

        a_ik = a_ik.view(K*batch_size, -1)
        a_ik = a_ik.view(K, batch_size, N)


        K_y = K_y.view(K, batch_size, N, N)
        k_ij = torch.softmax(K_y, dim=0)


        t_k = t_y.view(K * batch_size, -1)
        t_k = t_k * (1 - path_mod)
        # When the packets reach destination node, don't transmit anymore i.e. make T_j to be 0
        t_k[t_k == 0] = -1e10

        t_k = t_k.view(K, batch_size, N)
        t_j = torch.softmax(t_k, dim=0)         ######## Second Constraint Implicit Satisfaction  ########

        R_t = torch.transpose(R, dim0=3, dim1=2)

        return k_ij, a_ik, t_j, R, R_t



def get_neighbors(node_idx, adj_matrix):
    neighbors = torch.nonzero(adj_matrix[node_idx], as_tuple=False).squeeze()
    if neighbors.numel() == 0:
        print("No neighbors found")
        return torch.tensor([], dtype=torch.long)
    return neighbors if neighbors.dim() > 0 else neighbors.unsqueeze(0)



def train_eval(epochs, model, batch_size, dual_step, xi):
    pbar = tqdm(range(epochs), desc=f"Training for n = {N} nodes and k = {K} flows")
    # Load the data
    loader = create_graph_data(random_seed, num_samples, N, K, T, batch_size, dc)
    all_epoch_results = defaultdict(list)
    for epoch in pbar:
        print('epoch= ', epoch)
        for phase in loader:
            if phase == 'train':
                print("\nEntered Training Phase")
                model.train()
            else:
                print("\nEntered Evaluation Phase")
                model.eval()

            all_variables = defaultdict(list)
            for data in loader[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    all_Kij = []
                    all_Kji = []
                    all_aik = []
                    all_Tk = []
                    all_At = []
                    all_A0 = []

                    avg_Kij = []
                    avg_Kji = []
                    avg_aik = []
                    avg_Tk = []
                    avg_info = []
                    avg_quelen = []

                    all_info = []
                    all_queue = []

                    q_ik = torch.zeros(K * batch_size * N, 1, device=device_arg)

                    if phase == 'train':
                        mu_dual = 1 * torch.rand(K * batch_size * N, 1, device=device_arg)
                    else:
                        mu_dual = torch.zeros(K * batch_size * N, 1, device=device_arg)

                    test_mu_t = []
                    phi_t = []
                    queue_t = []
                    a_t = []
                    store_aik = []
                    cv1 = []

                    # Assuming each node has a state vector that indicates whether it has a packet for each destination k
                    node_states = torch.zeros((K, batch_size, N), device=device_arg)
                    reception_threshold = 0.01  # Define based on your domain knowledge, e.g., a function of distance, SNR, etc.
                    broadcast_threshold = 0.01

                    # Initialize a dictionary to accumulate packet counts for each flow and batch
                    node_packet_count_over_time = []

                    # Initialize a tensor of dictionaries to store transmitted and received packet IDs for each node:
                    packet_ids = [[[set() for _ in range(N)] for _ in range(batch_size)] for _ in range(K)]  # or initialize an appropriate tensor
                    transmitted_packet_ids = [[[set() for _ in range(N)] for _ in range(batch_size)] for _ in range(K)]
                    received_packet_ids = [[[set() for _ in range(N)] for _ in range(batch_size)] for _ in range(K)]
                    buffer_packet_ids = [[[set() for _ in range(N)] for _ in range(batch_size)] for _ in range(K)]
                    packet_ids = {}  # or initialize an appropriate tensor
                    transmitted_packet_ids = {}
                    received_packet_ids = {}
                    buffer_packet_ids = {}
                    for k in range(K):
                        for b in range(batch_size):
                            for n in range(N):
                                packet_ids[
                                    (k, b, n)] = set()  # Each entry is a set to store unique packet IDs for each packet
                                transmitted_packet_ids[
                                    (k, b, n)] = set()  # Each entry is a set to store unique transmitted packet IDs
                                received_packet_ids[
                                    (k, b, n)] = set()  # Each entry is a set to store unique received packet IDs
                                buffer_packet_ids[
                                    (k, b, n)] = set()  # Each entry is a set to store unique received packet IDs for the buffer

                    path_updated = torch.zeros((K, batch_size, N), device=device_arg)

                    for t in range(T):
                        # print("t = ", t)
                        data = data.to(device_arg)
                        # Load the packet identification tensors
                        A0_t = data.x
                        flow_num_t = A0_t['flow_num'][:, :, t].T
                        source_node_t = A0_t['source_node'][:, :, t].T
                        packet_num_t = A0_t['packet_num'][:, :, t].T
                        time_step_t = A0_t['time_step'][:, :, t].T

                        flow_num_t = flow_num_t.view(K, batch_size, N)
                        source_node_t = source_node_t.view(K, batch_size, N)
                        packet_num_t = packet_num_t.view(K, batch_size, N)
                        time_step_t = time_step_t.view(K, batch_size, N)


                        A0_t = packet_num_t
                        A0_t = A0_t.reshape(-1, 1)

                        temp_mat = data.network[:,:,t].T
                        net_graph = temp_mat.cpu().numpy()
                        adj_mat = temp_mat.view(N, batch_size, N)

                        # Create a networkx graph from the adjacency matrix
                        network_plot = nx.Graph(adj_mat[:,0,:].cpu().numpy()) # Make sure to update this when you have
                        # multiple samples in a batch

                        node_pos = data.pos[:, :, t]
                        node_pos = node_pos.cpu().numpy()

                        # Create a dictionary where the keys are node IDs and the values are the corresponding positions
                        pos_dict = {i: node_pos[i].tolist() for i in range(node_pos.shape[0])}

                        # Set the positions as node attributes in your NetworkX graph
                        nx.set_node_attributes(network_plot, pos_dict, 'pos')

                        tensor_eig = torch.tensor(data.eig)
                        single_eig = tensor_eig[:, t]
                        single_eig = single_eig.reshape(-1, 1)

                        single_flow_edge_index = data.edge_index[t].to(torch.int64)
                        single_flow_edge_weight = data.edge_attr[t].to(torch.float)


                        edge_index = []
                        edge_weight = []
                        eigen = []
                        for flow in range(K):
                            flow_edge_index = single_flow_edge_index + (flow * batch_size * N)
                            edge_index.append(flow_edge_index)

                            flow_eigen = single_eig  # + (flow * batch_size * N)
                            eigen.append(flow_eigen)

                            flow_edge_weight = single_flow_edge_weight  # + (flow * batch_size * N)
                            edge_weight.append(flow_edge_weight)

                        # print('After concatenating')
                        edge_index = torch.cat(edge_index, dim=1)
                        edge_weight = torch.cat(edge_weight, dim=0)
                        eigen = torch.cat(eigen, dim=0)

                        path_init = data.map[:, :, t].T
                        path_init = path_init.reshape(-1, 1)

                        buffer_packet_ids = {}
                        for k in range(K):
                            for b in range(batch_size):
                                for n in range(N):
                                    buffer_packet_ids[(k, b, n)] = set()  # Each entry is a set to store
                                    # unique received packet IDs for the buffer

                        if t == 0: # At t=0 since no packets have been received yet
                            A_t = A0_t
                            node_packet_count = A0_t
                            path = path_init
                            path_updated = path_init

                            # # Prepare packet IDs for the first time step
                            for k in range(K):
                                for b in range(batch_size):
                                    for source_node in range(N):
                                        packet_count = packet_num_t[k, b, source_node].item()
                                        if packet_count > 0:
                                            packet_id = f"{k:02d}{source_node:02d}{packet_count:02d}{t:03d}"
                                            # print("Packet ID:", packet_id)
                                            packet_ids[(k, b, source_node)] = {packet_id}

                        else:
                            # It should be updated at the end of the propagation step in the previous time t-1
                            A_t = A0_t + node_packet_count.reshape(-1, 1)

                            # Prepare relayed packet IDs for the next time step
                            for k in range(K):
                                for b in range(batch_size):
                                    # Identify the source and destination nodes for the flow
                                    source_nodes = torch.nonzero(path[k, b] == -1, as_tuple=True)[0]
                                    destination_node = torch.argmax(path[k, b]).item()
                                    for node in range(N):
                                        # Update packet_ids with new packet IDs
                                        packet_count = packet_num_t[k, b, node].item()
                                        # print(packet_count)
                                        if packet_count > 0 and node != destination_node:
                                            new_packet_id = f"{k:02d}{node:02d}{packet_count:02d}{t:03d}"
                                            packet_ids[(k, b, node)].add(new_packet_id)

                                        # Add relayed packet IDs
                                        if node != destination_node:  # Check if node is not the destination
                                            relayed_packet_ids = received_packet_ids[(k, b, node)]
                                            for relayed_packet_id in relayed_packet_ids:
                                                packet_ids[(k, b, node)].add(relayed_packet_id)
                                                # print("Print packet_ids inside relayed packets loop")
                                                # print(packet_ids)

                            path = path_updated.reshape(-1, 1)


                        A_t_normalized, A_t_mean, A_t_std = normalize_features(A_t.float())
                        A0_t_normalized, A0_t_mean, A0_t_std = normalize_features(A0_t.float())


                        x = torch.cat((A_t_normalized, mu_dual, path), dim=1)

                        # Call the GNN model
                        K_ij, a_ik, T_k, R_ij, R_ji = model(x, edge_index, edge_weight, K, batch_size, N, eigen,
                                                                  A0_t_normalized, device_arg)

                        a_ik = ((a_ik * A_t_std) + A_t_mean).round().float()

                        ########## PACKET COMMUNICATION SECTION ###########

                        ######### SECTION 1: TRANSMISSION #########

                        ########## PACKET BROADCAST BY TRANSMITTING NODES #############
                        # Simulate packet broadcast by node
                        path = path.view(K, batch_size, N)

                        # Broadcasting decision based on T_i^k
                        # T_i^k is the GNN output for transmission decision
                        broadcast_decision = T_k > broadcast_threshold


                        source_nodes = (path == -1)#.nonzero()
                        broadcast_success = torch.zeros_like(T_k)

                        # Update broadcast_success for source nodes with a successful broadcast decision
                        broadcast_success = broadcast_success.masked_fill(source_nodes & broadcast_decision, 1)


                        # # Assuming packet IDs are stored in A_t and are integers
                        for k in range(K):
                            for b in range(batch_size):
                                for node in range(N):
                                    # Check if this node is a source node for this flow and batch
                                    if broadcast_success[k, b, node]:
                                        # if source_nodes[k, b, node] and broadcast_decision[k, b, node]:
                                        #     broadcast_success[k, b, node] = 1
                                        # Iterate through each packet ID stored for this node
                                        for packet_id in packet_ids[(k, b, node)]:
                                            transmitted_packet_ids[k, b, node].add(packet_id) # Store the transmitted packet ID
                                            # Extract the time step from the packet ID
                                            packet_time_step = int(packet_id[6:9])
                                            # Check if the packet's time step matches the current time step
                                            if packet_time_step == t:
                                                # Add only current timestep packets to buffer_packet_ids
                                                buffer_packet_ids[k, b, node].add(packet_id)
                                            else:
                                                # For packets from previous time steps, add only if they were received in the last time step
                                                if packet_id in received_packet_ids[(k, b, node)]:
                                                    buffer_packet_ids[k, b, node].add(packet_id)

                        # Now transmitted_packet_ids_tensor contains the IDs of transmitted packets

                        ######### SECTION 2: RECEPTION #########

                        ########## PACKET RECEPTION BY NEIGHBORING/RECEIVING NODES #############
                        # Simulate packet propagation to neighboring nodes
                        node_states = node_states > 0  # Ensure node_states is a boolean tensor
                        new_states = broadcast_success.unsqueeze(3) * R_ij
                        new_states = new_states > reception_threshold


                        # Updating node states
                        node_states |= new_states.any(dim=2)

                        # Update the number packets at each node for each flow
                        node_packet_t = node_states.float()
                        node_packet_count = node_packet_count.view(K, batch_size, N)
                        node_packet_count = node_packet_count.float()  # Convert to Float
                        node_packet_count_over_time.append(node_packet_t)

                        # Calculate the number of packets received at each node after a transmitting node broadcasts the packets
                        path = path.view(K, batch_size, N)

                        # Loop over each flow and batch
                        for k in range(K):  # For each flow
                            for b in range(batch_size):
                                sources = torch.nonzero(path[k, b, :] == -1, as_tuple=True)[0]
                                dest_node = torch.argmax(path[k, b, :]).item()  # Determine the source node for flow k
                                for source_node in sources:
                                    # print(
                                    #     f"For flow {k}, Batch {b}: the source is Node {source_node} and destination is Node {dest_node}")
                                    current_adj_matrix = adj_mat[:, b, :]

                                    neighbors = get_neighbors(source_node.item(), current_adj_matrix)

                                    for j in neighbors:
                                        # Check if neighbor j received the packet
                                        if node_states[k, b, j] and K_ij[k,b,source_node.item(), j.item()] > reception_threshold:
                                            # Extract packet ID for the current packet
                                            packet_id = buffer_packet_ids[(k, b, source_node.item())]
                                            # print(
                                            #     f"Processing packet {packet_id} from Source {source_node.item()} to Neighbor {j.item()}")
                                            for packet_id in buffer_packet_ids[(k, b, source_node.item())]:
                                                source_id = int(packet_id[2:4])  # Extract the source node for the current packet
                                                if packet_id not in received_packet_ids[
                                                    k, b, j.item()] and j.item() != source_id:
                                                    received_packet_ids[k, b, j.item()].add(packet_id)
                                                    node_packet_count[k, b, j.item()] += 1

                                                    # print(
                                                    #     f"Updated received_packet_ids for Flow {k}, "
                                                    #     f"Neighbor {j.item()}: {received_packet_ids[k, b, j.item()]} from source node {source_node}")

                                                # else:
                                                    # This is a duplicate packet, so discard it
                                                    # print(
                                                        # f"Flow {k}: Neighbor {j} received duplicate packet from source "
                                                        # f"node {source_node}, dropping it now")
                                                    # continue


                        ########### UPDATE PATH TO FIND THE TRANSMITTING NODES FOR NEXT TIME STEP - Alt code  ##########

                        # After updating node_states and node_packet_count:
                        path_updated = path_updated.view(K, batch_size, N)

                        # Determine the destination nodes for each flow
                        dest_nodes = torch.argmax(path, dim=2)

                        # Get nodes that received packets, which are not the destination
                        receivers = (node_states > 0).nonzero(as_tuple=False)

                        # Create a mask to exclude destination nodes
                        mask = receivers[:, 2] != dest_nodes[receivers[:, 0], receivers[:, 1]]

                        # Apply the mask to get the valid receivers
                        valid_receivers = receivers[mask]

                        # Update the path_updated tensor
                        path_updated[valid_receivers[:, 0], valid_receivers[:, 1], valid_receivers[:, 2]] = -1


                        # Evaluate the objective function or loss function
                        A0_t = A0_t.view(K, batch_size, -1)
                        phi = torch.sum((torch.sum(torch.log(a_ik + 10e-10), dim=0)), dim=-1)

                        path_out = path_init.view(K * batch_size, N)
                        path_out = path_out.view(K, batch_size, N)

                        # Create a mask to exclude terms where j == i
                        mask = torch.ones(N, N, device=device_arg) - torch.eye(N, device=device_arg)

                        K_ij = K_ij.view(K * batch_size, N, N)
                        R_ij = R_ij.view(K * batch_size, N, N)
                        T_q = T_k.unsqueeze(3)
                        T_q = T_q.view(K * batch_size, N, 1)

                        # Evaluate the queue length
                        q_ik.data = torch.relu(q_ik.view(K * batch_size * N, -1) + A0_t.view(K * batch_size * N, -1) + \
                            torch.bmm((R_ij * K_ij * mask), T_q).view(K * batch_size * N, -1) * a_ik.view(K * batch_size * N, -1)
                            - (T_q.view(K * batch_size * N, -1) * C.view(K * batch_size * N, -1)))

                        q_ik = torch.squeeze(q_ik)
                        q_ik = torch.round(q_ik)

                        q_ik = torch.squeeze(q_ik)
                        q_ik = q_ik.view(K, batch_size, N)
                        q_len = torch.sum(torch.sum(q_ik, dim=0), dim=-1)


                        all_Kij.append(K_ij)
                        all_aik.append(a_ik)
                        all_info.append(phi)
                        all_Tk.append(T_k)
                        all_queue.append(q_len)
                        all_At.append(A_t)
                        all_A0.append(A0_t)

                        phi_t.append(phi.detach().cpu())
                        queue_t.append(q_ik.view(K * batch_size * N).detach().cpu())
                        a_t.append(A_t.view(K * batch_size * N).detach().cpu())

                        if phase != 'train':
                            if (t + 1) % T0 == 0:
                                test_Kij = torch.stack(all_Kij[-T0:], dim=0)
                                test_aik = torch.stack(all_aik[-T0:], dim=0)
                                test_Tk = torch.stack(all_Tk[-T0:], dim=0)
                                test_At = torch.stack(all_At[-T0:], dim=0)

                                recent_Kij = torch.mean(test_Kij, dim=0)
                                recent_aik = torch.mean(test_aik, dim=0)
                                recent_Tk = torch.mean(test_Tk, dim=0)
                                recent_At = torch.mean(test_At, dim=0)

                                recent_aik = recent_aik.view(K * batch_size, -1)
                                recent_aik = recent_aik.view(K * batch_size * N, -1)

                                xi = xi.view(K * batch_size * N, -1)

                                recent_Kij = recent_Kij.view(K, batch_size, N, N)

                                T_j = recent_Tk.unsqueeze(3)


                                # Create a mask to exclude terms where j == i
                                mask = torch.ones(N, N, device=device_arg) - torch.eye(N, device=device_arg)

                                sum_j = torch.bmm((mask * R_ij * recent_Kij.view(K * batch_size, N, N)),
                                                  T_j.view(K * batch_size, N, 1)) * recent_aik.view(K * batch_size, N,
                                                                                                    -1)

                                sum_j = torch.squeeze(sum_j).view(K * batch_size * N, -1)

                                # Update the dual variables using the Method of Multipliers
                                mu_dual.data = torch.relu(mu_dual +
                                                          dual_step * ((T_j.view(K * batch_size * N, -1) *
                                                                        C.view(K * batch_size * N, -1)) - sum_j - recent_aik  - xi))


                                constr_vlt = ((T_j.view(K * batch_size * N, -1) * C.view(K * batch_size * N, -1)) -
                                              sum_j - recent_aik)


                                test_mu_t.append(mu_dual.detach().cpu())
                                store_aik.append(recent_aik.view(K, batch_size, N))
                                cv1.append(constr_vlt)

                    if phase != 'train':
                        test_mu_t = torch.stack(test_mu_t, dim=0)
                        queue_t = torch.stack(queue_t, dim=0)
                        phi_t = torch.stack(phi_t, dim=0)
                        a_t = torch.stack(a_t, dim=0)
                        cv1 = torch.stack(cv1, dim=0)

                    all_Kij = torch.stack(all_Kij, dim=0)
                    # all_Kji = torch.stack(all_Kji, dim=0)
                    all_aik = torch.stack(all_aik, dim=0)
                    all_Tk = torch.stack(all_Tk, dim=0)
                    all_info = torch.stack(all_info, dim=0)
                    all_queue = torch.stack(all_queue, dim=0)
                    all_At = torch.stack(all_At, dim=0)
                    all_A0 = torch.stack(all_A0, dim=0).float()
                    # all_A0 = torch.tensor(all_A0, dtype=torch.float32)

                    avg_Kij = torch.mean(all_Kij, dim=0)
                    # avg_Kji = torch.mean(all_Kji, dim=0)
                    avg_aik = torch.mean(all_aik, dim=0)
                    avg_Tk = torch.mean(all_Tk, dim=0)
                    avg_At = torch.mean(all_At, dim=0)
                    avg_A0 = torch.mean(all_A0, dim=0)

                    avg_info = torch.mean(all_info, dim=0)
                    avg_quelen = torch.mean(all_queue, dim=0)

                    node_packet_average = node_packet_count / T

                    if phase == 'train':
                        mu_dual = torch.squeeze(mu_dual)
                        mu_dual = mu_dual.view(K, batch_size * N)
                        mu_dual = mu_dual.view(K, batch_size, N)
                        xi = xi.view(K, batch_size, N)

                        avg_Kij = avg_Kij.view(K, batch_size, N, N)

                        avg_aik = avg_aik.view(K, batch_size, N)
                        avg_Tk = avg_Tk.view(K, batch_size, N)

                        U = torch.sum((torch.sum(torch.log(avg_aik + 10e-10), dim=0)), dim=-1)


                        T_j = avg_Tk.unsqueeze(3)

                        # Create a mask to exclude terms where j == i
                        mask = torch.ones(N, N, device=device_arg) - torch.eye(N, device=device_arg)

                        sum_j_train = torch.bmm((mask * R_ij * avg_Kij.view(K * batch_size, N, N)),
                                                T_j.view(K * batch_size, N, 1)) * avg_aik.view(K * batch_size, N, -1)

                        sum_j_train = torch.squeeze(sum_j_train).view(K, batch_size, N)

                        T2 = mu_dual * ((avg_Tk * C.view(K, batch_size, N)) - avg_aik - sum_j_train - xi)


                        T3 = avg_Tk * C.view(K, batch_size, N) - avg_aik - sum_j_train - xi


                        # Compute the augmented Lagrangian for the Method of Multipliers
                        L = -(U - torch.sum(torch.sum(T2 + 0.5 * dual_step * torch.square(T3), dim=0), dim=-1)).mean()

                        L.backward()
                        # Optimize the primal and dual variable for the Method of Multipliers using the Adam Optimizer
                        optimizer.step()

                all_variables['info'].extend(avg_info.detach().cpu().numpy().tolist())
                all_variables['queue'].extend(avg_quelen.detach().cpu().numpy().tolist())
                all_variables['queue_along_time'] = torch.mean(all_queue, dim=1).detach().cpu().numpy().tolist()
                all_variables['phi_along_time'] = torch.mean(all_info, dim=1).detach().cpu().numpy().tolist()

                if phase != 'train':
                    all_variables['mu_over_time'].append(test_mu_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['test_mu_over_time'].append(
                        torch.mean(test_mu_t.squeeze(-1).T, dim=0).detach().cpu().numpy())

                    all_variables['average_queue_over_time'].append(
                        torch.mean(queue_t.squeeze(-1).T, dim=0).detach().cpu().numpy())
                    all_variables['queue_over_time'].append(queue_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['phi_over_time'].append(phi_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['a_over_time'].append(a_t.squeeze(-1).T.detach().cpu().numpy())

                    all_variables['all_info'].append(all_info.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['all_queue_len'].append(all_queue.squeeze(-1).T.detach().cpu().numpy())

            scheduler.step(L)

            for key in all_variables:
                if key == 'info':
                    all_epoch_results[phase, 'info_mean'].append(np.mean(all_variables['info']))
                    all_epoch_results[phase, 'info_max'].append(np.max(all_variables['info']))

                elif key == 'queue':
                    all_epoch_results[phase, 'queue_mean'].append(np.mean(all_variables['queue']))

                elif key in ['test_mu_over_time', 'mu_over_time']:
                    all_epoch_results[phase, 'test_mu_over_time'] = all_variables['test_mu_over_time']
                    all_epoch_results[phase, 'mu_over_time'] = all_variables['mu_over_time']

                elif key in ['queue_over_time']:
                    all_epoch_results[phase, 'queue_over_time'] = all_variables['queue_over_time']

                elif key in ['phi_over_time']:
                    all_epoch_results[phase, 'phi_over_time'] = all_variables['phi_over_time']

                elif key in ['a_over_time']:
                    all_epoch_results[phase, 'a_over_time'] = all_variables['a_over_time']

                elif key in ['average_queue_over_time']:
                    all_epoch_results[phase, 'average_queue_over_time'] = all_variables['average_queue_over_time']

                elif key in ['phi_along_time', 'queue_along_time']:
                    all_epoch_results[phase, 'phi_along_time'] = all_variables['phi_along_time']
                    all_epoch_results[phase, 'queue_along_time'] = all_variables['queue_along_time']

                else:
                    all_epoch_results[phase, key].append(np.mean(all_variables[key]))

    return all_epoch_results, network_plot, node_packet_average, node_packet_count_over_time



random_seed = 1357537
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

N = 10
T = 100
T0 = 5
K = 4
dc = 0.3
device_arg = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"

batch_Size = 16  # 8#16  # 8#4#2#32 #16 #2 #32
nEpochs = 25 #30#25#30  # 40#50#100#20#30#50#100 #100
nTrain = 128 #256 # 128#64#32#16#128 #64 #128 #16
nTest = 16  # 16  # 8#4#2#32 #16 #32 #64
num_samples = {'train': nTrain, 'eval': nTest}
hk_primal_step = 0.05
mu_dual_step = 0.005

last_feature = 8
num_features_list = [3] + [16] + [last_feature]
flow_k = 0

# Initialize th auxiliary variable for Method of Multipliers
xi_aux = torch.rand(K * batch_Size * N, 1, device=device_arg)

# Initialize the node capacities
cap = 500
C = cap * torch.ones(K * batch_Size * N, 1, device=device_arg)

gnn_model = GNN_Model(num_features_list).to(device_arg)
optimizer = torch.optim.Adam(list(gnn_model.parameters()) + [xi_aux], hk_primal_step)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1)


results, net_plot, avg_node_packt, node_pkt_time = train_eval(nEpochs, gnn_model, batch_Size, mu_dual_step,
                                                              xi_aux)
features_str = '_'.join(map(str, num_features_list))

# Define the directory path
save_dir = "/home/sourajit/GNNWirelessComm/models/slack/"
os.makedirs(save_dir, exist_ok=True)
filename = os.path.join(save_dir, 'SA_Model_' + str(N) + 'N_' +
                str(K) + 'K_' + features_str + 'lyrs_' + str(nTrain) + 'S' + str(cap) + 'C.pt')

torch.save(gnn_model.state_dict(), filename)

mean_info = results['train', 'info_mean']
mean_info_eval = results['eval', 'info_mean']

max_info_eval = results['eval', 'info_max']

mean_queue_train = results['train', 'queue_mean']
mean_queue_eval = results['eval', 'queue_mean']
queue_ind = results['eval', 'queue_over_time']
phi_ind = results['eval', 'phi_over_time']
mu_ind = results['eval', 'mu_over_time']
a_ind = results['eval', 'a_over_time']

mu_plot = results['eval', 'test_mu_over_time']
queue_plot = results['eval', 'average_queue_over_time']

info_along = results['eval', 'phi_along_time']
queue_along = results['eval', 'queue_along_time']

np.savez(f'SA_{N}nodes_{K}flows_TrTs_{hk_primal_step}p_{mu_dual_step}d_{nEpochs}e.npz',
         all=results, sa_a=mean_info_eval, sa_a_max=max_info_eval, sa_a_train=mean_info,
         sa_q=mean_queue_eval, mu=mu_plot, mu_ind=mu_ind, queue_ind=queue_ind, phi_ind=phi_ind, a_ind=a_ind)

img_path = '/home/sourajit/GNNWirelessComm/Exor_Problem2/State Augmentation/Slack/Expt59/'
os.makedirs(img_path, exist_ok=True)
timestamps = datetime.now()

plt.figure(0)
# plt.figure(0)
plt.subplot(2, 1, 1)
plt.plot(mean_info_eval, label='Evaluation Utility')
plt.title("GNN+MoM: Evaluation Utility for N=%d nodes, K=%d flows" % (N, K) + features_str)
plt.xlabel("Epochs")
plt.ylabel("Mean value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2, 1, 2)
plt.plot(mean_queue_eval, label='Evaluation Queue length')
plt.title("Queue length for primal step=%0.4f, dual step=%0.5f" % (hk_primal_step, mu_dual_step))
plt.xlabel("Epochs")
plt.ylabel("Max value")
plt.legend()
plt.grid()
plt.tight_layout()


plt.savefig(img_path + 'SA_Exor_Obj_MoM' + timestamps.strftime("%Y-%m-%d_%H-%M-%S") + '.png', dpi=300,
            bbox_inches="tight")


plt.figure(1)
plt.subplot(2, 1, 1)
for i in range(K * nTest * N):
    plt.plot(mu_ind[0][i])
plt.title("Dual variables, mu for T=%d, N=%d, K=%d and layers" % (T, N, K) + features_str)
plt.xlabel("T/T0")
plt.ylabel("Max value")
plt.grid()
plt.tight_layout()
plt.subplot(2, 1, 2)
for i in range(K * batch_Size * N):
    plt.plot(queue_ind[0][i])
plt.title("Queue lengths for T0=%d, primal step=%0.4f and dual step=%0.5f" % (T0, hk_primal_step, mu_dual_step))
plt.xlabel("T/T0")
plt.ylabel("Max value")
plt.grid()
plt.tight_layout()
plt.savefig(img_path + 'Statistics_Obj_MoM' + timestamps.strftime("%Y-%m-%d_%H-%M-%S") + '.png', dpi=300,
            bbox_inches="tight")

plt.show()