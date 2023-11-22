from routing_utils import *
from geopy import distance
import pandas as pd

def fixed_radius_sampling(center, radius, nb_samples):
    # center      is a tuple of the coordinates of the center of the circles
    # radius      is the radius to sample on (in km)
    # nb_samples   is the number of samples on each circle

    res = []

    for theta in range(nb_samples):
        point = distance.distance(kilometers=radius).destination(center, theta * (360/nb_samples))
        res = res + [(point[0], point[1])]

    # return a list of tuples with the coordinates of the samples
    return res

def fixed_radius(net, center, radius, nb_samples):
    dfs = []
    od_pairs = []

    for r in radius:
        df = pd.DataFrame(columns=['coor', 'edge'])
        samples = fixed_radius_sampling(center, r, nb_samples)

        for s in samples:
                sumo_coor = net.convertLonLat2XY(s[1], s[0])
                candidates_edges = net.getNeighboringEdges(sumo_coor[0], sumo_coor[1], r=200)

                # We check whether there is edges that are reasonably close from our points
                if candidates_edges != []:
                    edge = sorted(candidates_edges, key = lambda x: x[1])[0][0].getID()

                    df.loc[len(df)] = [s, edge]

        dfs = dfs + [df]

    for df in dfs:
        edges = df['edge']

        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                od_pairs = od_pairs + [(edges[i], edges[j])]
                od_pairs = od_pairs + [(edges[j], edges[i])]

    return od_pairs
