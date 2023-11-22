from haversine import haversine
from routing_utils import *
from geopy import distance
import pandas as pd
import math

def fixed_radius_sampling(center, radius, nb_samples):
    # center      is the coordinate of the city center
    # radius      is a list of different radius to sample on
    # nb_samples   is the number of samples on a circle

    res = []

    for theta in range(nb_samples):
        point = distance.distance(kilometers=radius).destination(center, theta * (360/nb_samples))
        res = res + [(point[0], point[1])]

    return res

def fixed_radius_OD_pairs(net, center, radius, nb_samples):
    df = pd.DataFrame(columns=['radius', 'euclidean', 'theta',
                               'DI_shortest', 'DI_fastest'])

    graph = from_sumo_to_igraph_network(net)

    for rad in radius:
        rad_samples = fixed_radius_sampling(center, rad, nb_samples)

        for i in range(len(rad_samples)):
            for j in range(i+1, len(rad_samples)):
                sumo_a = net.convertLonLat2XY(rad_samples[i][1], rad_samples[i][0])
                sumo_b = net.convertLonLat2XY(rad_samples[j][1], rad_samples[j][0])

                euc = haversine(rad_samples[i], rad_samples[j])

                candidates_edges_a = net.getNeighboringEdges(sumo_a[0], sumo_a[1], r=200)
                candidates_edges_b = net.getNeighboringEdges(sumo_b[0], sumo_b[1], r=200)

                # We check weather there is edges that are reasonably close from our points
                if(candidates_edges_a != [] and candidates_edges_b != []):
                    edge_a = sorted(candidates_edges_a, key = lambda x: x[1])[0][0].getID()
                    edge_b = sorted(candidates_edges_b, key = lambda x: x[1])[0][0].getID()

                    shortest_path_ab = get_shortest_path(graph, edge_a, edge_b, 'length')['cost']
                    shortest_path_ba = get_shortest_path(graph, edge_b, edge_a, 'length')['cost']

                    # We make sure that euc/2 is lower than the radius
                    # The approximation could create a math domain error in the math.asin if not
                    if euc/2 > rad:
                        euc = rad/2

                    # We check weather there exists a path from a to b and from b to a
                    # We also verify that its length is greater than the euclidean distance (with the approximation of the closest edge it could be problematic otherwise)
                    if shortest_path_ab > euc and shortest_path_ab < math.inf:
                        fp_ab = get_shortest_path(graph, edge_a, edge_b, 'traveltime')['ig']
                        fastest_path_ab = compute_path_cost(graph, fp_ab, 'length')
                        df.loc[len(df)] = [rad, euc, abs(2 * math.asin((euc/2) / rad) / math.pi),
                                           (shortest_path_ab / 1000) / euc, (fastest_path_ab / 1000) / euc]

                    if shortest_path_ba > euc and shortest_path_ba < math.inf:
                        fp_ba = get_shortest_path(graph, edge_b, edge_a, 'traveltime')['ig']
                        fastest_path_ba = compute_path_cost(graph, fp_ba, 'length')
                        df.loc[len(df)] = [rad, euc, abs(2 * math.asin((euc/2) / rad) / math.pi),
                                           (shortest_path_ba / 1000) / euc, (fastest_path_ba / 1000) / euc]

    return df
