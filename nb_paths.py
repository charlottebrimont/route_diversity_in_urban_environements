from shapely.affinity import scale, rotate
from routing_algorithms import *
from path_diversity import *
from routing_utils import *
from tqdm import tqdm

import pandas as pd
import shapely
import random
import math

def generate_ellipse(start, end):
    d = shapely.Point(start[1], start[0]).distance(shapely.Point(end[1], end[0]))
    center = shapely.Point((start[1]+end[1])/2, (start[0]+end[0])/2)

    # alpha represents the angle of this rotation
    alpha = math.atan2(end[1]-start[1], end[0]-start[0])

    # creating a circle polygon
    # scaling the y axis to obtain an ellipse shape
    # rotating it to orientate according to the OD points
    ellipse = center.buffer(d/2)
    ellipse = scale(ellipse, yfact=math.sqrt(2))
    ellipse = rotate(ellipse, -1 * alpha, origin=center, use_radians=True)

    return ellipse

def elliptic_subgraph(graph, ori, des):
    sub_vs=set()
    ellipse=generate_ellipse(ori, des)

    for edge in graph.es:
        to_lon, to_lat = edge['coordinates']['to']
        to_p = shapely.Point(to_lon, to_lat)

        from_lon, from_lat = edge['coordinates']['from']
        from_p = shapely.Point(from_lon, from_lat)

        if shapely.contains(ellipse, to_p) and shapely.contains(ellipse, from_p):
            sub_vs.add(edge.source)
            sub_vs.add(edge.target)

    return graph.subgraph(sub_vs)

def approximated_simple_paths(G, from_edge, to_edge, it=1, nb_paths=100, delta=0, tau=100, attribute='length'):
    simple_paths=set()

    og_attribute = G.es[attribute]

    for i in range(it):
        #Graph randomization
        paths = graph_randomization(G, from_edge, to_edge, 1, delta, tau, attribute, remove_tmp_attribute=False)
        for p in paths:
            simple_paths.add(tuple(p['ig']))

        G.es[attribute] = G.es[f"tmp_{attribute}"]

        #Path penalization
        paths = path_penalization(G, from_edge, to_edge, nb_paths, 1, attribute, remove_tmp_attribute=False)
        for p in paths:
            simple_paths.add(tuple(p['ig']))

        G.es[attribute] = og_attribute

    return simple_paths

def od_matrix_paths(G, od_pairs, nb_samples, epsilon=[], it=1, nb_paths=100, delta=.5, tau=1):
    df_paths = pd.DataFrame(columns=['origin', 'destination', 'euclidean', 'paths'])

    idx = random.sample(range(0, len(od_pairs)), nb_samples)

    for i in tqdm(idx):
        # Get the data from od_data
        data = od_pairs[f'vehicle_{i}']
        index = data['edges']

        ig_ori = G.es.find(id=index[0])
        ig_des = G.es.find(id=index[1])

        ori = (ig_ori['coordinates']['from'][1], ig_ori['coordinates']['from'][0])
        des = (ig_des['coordinates']['to'][1], ig_des['coordinates']['to'][0])

        # Compute the ellipse
        subgraph = elliptic_subgraph(G, ori, des)

        # Test whether the edge is included in the subgraph
        try:
            ig_a = subgraph.es.find(id=index[0])
            ig_b = subgraph.es.find(id=index[1])
        except Exception:
            continue

        # Compute simple paths
        paths = list(approximated_simple_paths(subgraph, ig_ori['id'], ig_des['id'], it=it, nb_paths=nb_paths, delta=delta, tau=tau))

        # Add to df
        new_row = [index[0], index[1], haversine(ig_ori['coordinates']['from'], ig_des['coordinates']['to']), paths]
        df_paths = df_paths.append(pd.Series(new_row, index=df_paths.columns[:len(new_row)]), ignore_index=True)

        #Compute epsilon simple paths
        shortest = get_shortest_path(G, index[0], index[1], 'length')['cost']

        for eps in epsilon:
            treshold = shortest * (1+eps)

            epsilon_paths = list(filter(lambda x: compute_path_cost(G, x, 'length') <= treshold, paths))
            df_paths.at[len(df_paths)-1, f'nb_{round(eps, 1)}_paths'] = len(epsilon_paths)

    df_paths['nb_paths'] = df_paths['paths'].apply(len)

    return df_paths

def radial_paths(G, od_pairs, nb_samples, epsilon=[], it=1, nb_paths=100, delta=.5, tau=1):
    df_paths = pd.DataFrame(columns=['origin', 'destination', 'euclidean', 'paths'])

    idx = random.sample(range(0, len(od_pairs)), nb_samples)

    for i in tqdm(idx):
        # Get the data from od_pairs
        index = od_pairs[i]

        ig_ori = G.es.find(id=index[0])
        ig_des = G.es.find(id=index[1])

        ori = (ig_ori['coordinates']['from'][1], ig_ori['coordinates']['from'][0])
        des = (ig_des['coordinates']['to'][1], ig_des['coordinates']['to'][0])

        # Compute the ellipse
        subgraph = elliptic_subgraph(G, ori, des)

        # Test whether the edges are in the subgraph
        try:
            ig_a = subgraph.es.find(id=index[0])
            ig_b = subgraph.es.find(id=index[1])
        except:
            continue

        # Compute simple paths
        try:
            paths = list(approximated_simple_paths(subgraph, ig_ori['id'], ig_des['id'], it=it, nb_paths=nb_paths, delta=delta, tau=tau))
        except:
            continue

        # Add to df
        new_row = [index[0], index[1], haversine(ig_ori['coordinates']['from'], ig_des['coordinates']['to']), paths]
        df_paths = df_paths.append(pd.Series(new_row, index=df_paths.columns[:len(new_row)]), ignore_index=True)

        #Compute epsilon simple paths
        shortest = get_shortest_path(G, index[0], index[1], 'length')['cost']

        for eps in epsilon:
            treshold = shortest * (1+eps)

            epsilon_paths = list(filter(lambda x: compute_path_cost(G, x, 'length') <= treshold, paths))
            df_paths.at[len(df_paths)-1, f'nb_{round(eps, 1)}_paths'] = len(epsilon_paths)

    df_paths['nb_paths'] = df_paths['paths'].apply(len)

    return df_paths
