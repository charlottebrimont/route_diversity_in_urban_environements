from shapely.affinity import scale, rotate
from haversine import haversine
import matplotlib.pyplot as plt
from routing_utils import *
from geopy import distance
from pyproj import Geod
import pandas as pd
import warnings
import shapely
import math

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def fixed_radius_sampling(center, radius, nb_samples):
    # center      is the coordinate of the city center
    # radius      is a list of different radius to sample on
    # nb_samples   is the number of samples on a circle

    res = []

    for theta in range(nb_samples):
        point = distance.distance(kilometers=radius).destination(center, theta * (360/nb_samples))
        res = res + [(point[0], point[1])]

    return res


def DI_dataframe(net, center, radius, nb_samples):
    df = pd.DataFrame(columns=['radius', 'euclidean', 'theta',
                               'DI_shortest', 'DI_fastest'])

    graph = from_sumo_to_igraph_network(net)

    for rad in radius:
        rad_samples = fixed_radius_sampling(center, rad, nb_samples)

        for i in range(len(rad_samples)):
            # We initialize the first point by getting the closest edge
            point_a = rad_samples[i]
            sumo_a = net.convertLonLat2XY(point_a[1], point_a[0])
            candidates_edges_a = net.getNeighboringEdges(sumo_a[0], sumo_a[1], r=100)

            if candidates_edges_a!=[]:
                edge_a = sorted(candidates_edges_a, key = lambda x: x[1])[0][0].getID()

                for j in range(i+1, len(rad_samples)):
                    point_b = rad_samples[j]
                    sumo_b = net.convertLonLat2XY(rad_samples[j][1], rad_samples[j][0])
                    candidates_edges_b = net.getNeighboringEdges(sumo_b[0], sumo_b[1], r=100)

                    # We check weather there is edge that are reasonably close from our points
                    if candidates_edges_b != []:
                        edge_b = sorted(candidates_edges_b, key = lambda x: x[1])[0][0].getID()

                        # We now compute all the values that need to be added to the final dataframe
                        #   euclidean distance
                        euc = haversine(point_a, point_b)

                        #   angle
                        ac_x, ac_y = center[0]-point_a[0], center[1]-point_a[1]
                        bc_x, bc_y = center[0]-point_b[0], center[1]-point_b[1]

                        scalar_product = (ac_x*bc_x) + (ac_y*bc_y)

                        magnitude_ac = math.sqrt(ac_x ** 2 + ac_y ** 2)
                        magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

                        theta = math.acos(scalar_product / (magnitude_ac * magnitude_bc))

                        # shortest path
                        shortest_path_ab = get_shortest_path(graph, edge_a, edge_b, 'length')['cost']
                        shortest_path_ba = get_shortest_path(graph, edge_b, edge_a, 'length')['cost']


                        # We check weather there exists a path from a to b and from b to a
                        # We also verify that its length is greater than the euclidean distance (with the approximation of the closest edge it could be problematic otherwise)
                        if shortest_path_ab > euc and shortest_path_ab < math.inf:
                            fp_ab = get_shortest_path(graph, edge_a, edge_b, 'traveltime')['ig']
                            fastest_path_ab = compute_path_cost(graph, fp_ab, 'length')
                            df.loc[len(df)] = [rad, euc, theta/math.pi, (shortest_path_ab / 1000) / euc, (fastest_path_ab / 1000) / euc]

                        if shortest_path_ba > euc and shortest_path_ba < math.inf:
                            fp_ba = get_shortest_path(graph, edge_b, edge_a, 'traveltime')['ig']
                            fastest_path_ba = compute_path_cost(graph, fp_ba, 'length')
                            df.loc[len(df)] = [rad, euc, theta/math.pi, (shortest_path_ba / 1000) / euc, (fastest_path_ba / 1000) / euc]

    return df

def detour_index(city_name, center, radius, nb_samples):
    net = sumolib.net.readNet('data/road_network/'+city_name+'_road_network.net.xml')

    df = DI_dataframe(net, center, radius, nb_samples)

    print("# of samples by radius")
    for r in df.radius.unique():
        print('%s: %s' % (r, df.loc[df['radius']==r].shape[0]))

    # We round all the data to avoid having to sparse data due to infinite decomals
    df['ratio'] = df.euclidean / df.radius
    df['ratio'] = df['ratio'].divide(5).round(decimals=2).mul(5)
    df['theta'] = df['theta'].divide(2).round(decimals=2).mul(2)
    df['euclidean'] = df['euclidean'].round(decimals=2)

    # We save the dataframe to csv in case we want to use it later
    df.to_csv('data/OD_samples/'+city_name+'_DI.csv', index=False)

    DI_r(df, city_name)
    DI_theta(df, city_name)
    DI_d_r(df, city_name)
    DI_d(df, city_name)

    return df

def DI_r(df, city):
    DI_short = df.groupby('radius')['DI_shortest'].mean().to_numpy()
    DI_fast = df.groupby('radius')['DI_fastest'].mean().to_numpy()

    radius = df.radius.unique()

    plt.title('DI by radius in ' + city, size=16)
    plt.plot(radius, DI_short, marker='o')
    plt.plot(radius, DI_fast, marker='o')
    plt.legend(['shortest', 'fastest'])
    plt.ylabel('Detour Index')
    plt.xlabel('r')

    plt.savefig('results/DI/' + city + '_r.png')
    plt.show()

def DI_theta(df, city):
    DI = df.groupby(['radius', 'theta'])['DI_shortest', 'DI_fastest'].mean().reset_index(level=[0, 1])

    x_shortest, y_shortest = [], []
    x_fastest, y_fastest = [], []

    for rad in DI.radius.unique():
        DI_rad_s = np.transpose(DI.loc[DI['radius']==rad][['theta', 'DI_shortest']].to_numpy())
        x_shortest.append(DI_rad_s[0])
        y_shortest.append(DI_rad_s[1])
        DI_rad_f = np.transpose(DI.loc[DI['radius']==rad][['theta', 'DI_fastest']].to_numpy())
        x_fastest.append(DI_rad_f[0])
        y_fastest.append(DI_rad_f[1])

    fig = plt.figure(figsize=(17, 5))

    ax = fig.add_subplot(121)
    plt.title('DI of shortest path by angle for each radius in ' + city, size=16)
    for i in range(len(x_shortest)):
        plt.plot(x_shortest[i], y_shortest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('theta')
    plt.legend(DI.radius.unique())

    fig.add_subplot(122, sharex=ax, sharey=ax)
    plt.title('DI of fastest path by angle for each radius in ' + city, size=16)
    for i in range(len(x_fastest)):
        plt.plot(x_fastest[i], y_fastest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('theta')
    plt.legend(DI.radius.unique())

    plt.savefig('results/DI/' + city + '_theta.png')
    plt.show()

def DI_d_r(df, city):
    DI = df.groupby(['radius', 'ratio'])['DI_shortest', 'DI_fastest'].mean().reset_index(level=[0, 1])

    x_shortest, y_shortest = [], []
    x_fastest, y_fastest = [], []

    for rad in DI.radius.unique():
        DI_rad_s = np.transpose(DI.loc[DI['radius']==rad][['ratio', 'DI_shortest']].to_numpy())
        x_shortest.append(DI_rad_s[0])
        y_shortest.append(DI_rad_s[1])
        DI_rad_f = np.transpose(DI.loc[DI['radius']==rad][['ratio', 'DI_fastest']].to_numpy())
        x_fastest.append(DI_rad_f[0])
        y_fastest.append(DI_rad_f[1])

    fig = plt.figure(figsize=(17, 5))

    ax = fig.add_subplot(121)
    plt.title('DI of shortest path by ratio for each radius in ' + city, size=16)
    for i in range(len(x_shortest)):
        plt.plot(x_shortest[i], y_shortest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('d/r')
    plt.legend(DI.radius.unique())

    fig.add_subplot(122, sharex=ax, sharey = ax)
    plt.title('DI of fastest path by ratio for each radius in ' + city, size=16)
    for i in range(len(x_fastest)):
        plt.plot(x_fastest[i], y_fastest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('d/r')
    plt.legend(DI.radius.unique())

    plt.savefig('results/DI/' + city + '_d_r.png')
    plt.show()

def DI_d(df, city):
    DI = df.groupby(['radius', 'euclidean'])['DI_shortest', 'DI_fastest'].mean().reset_index(level=[0, 1])

    x_shortest, y_shortest = [], []
    x_fastest, y_fastest = [], []

    for rad in DI.radius.unique():
        DI_rad_s = np.transpose(DI.loc[DI['radius']==rad][['euclidean', 'DI_shortest']].to_numpy())
        x_shortest.append(DI_rad_s[0])
        y_shortest.append(DI_rad_s[1])
        DI_rad_f = np.transpose(DI.loc[DI['radius']==rad][['euclidean', 'DI_fastest']].to_numpy())
        x_fastest.append(DI_rad_f[0])
        y_fastest.append(DI_rad_f[1])

    fig = plt.figure(figsize=(17, 5))

    ax = fig.add_subplot(121)
    plt.title('DI of shortest path by euclidean distance in ' + city, size=16)
    for i in range(len(x_shortest)):
        plt.plot(x_shortest[i], y_shortest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('d')
    plt.legend(DI.radius.unique())

    fig.add_subplot(122, sharex=ax, sharey=ax)
    plt.title('DI of fastest path by euclidean distance in ' + city, size=16)
    for i in range(len(x_fastest)):
        plt.plot(x_fastest[i], y_fastest[i], marker='o')
    plt.ylabel('Detour Index')
    plt.xlabel('d')
    plt.legend(DI.radius.unique())

    plt.savefig('results/DI/' + city + '_d.png')
    plt.show()


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

def get_all_ellipses(center, radius, nb_samples):
    df = pd.DataFrame(columns=['start', 'end', 'radius', 'euclidean', 'theta', 'ellipse', 'vs', 'es'])

    for rad in radius:
        rad_samples = fixed_radius_sampling(center, rad, nb_samples)

        for i in range(len(rad_samples)):
            point_a = rad_samples[i]
            for j in range(i+1, len(rad_samples)):
                point_b = rad_samples[j]

                # euclidean distance
                euc = haversine(point_a, point_b)

                #   angle
                ac_x, ac_y = center[0]-point_a[0], center[1]-point_a[1]
                bc_x, bc_y = center[0]-point_b[0], center[1]-point_b[1]

                scalar_product = (ac_x*bc_x) + (ac_y*bc_y)

                magnitude_ac = math.sqrt(ac_x ** 2 + ac_y ** 2)
                magnitude_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

                theta = math.acos(scalar_product / (magnitude_ac * magnitude_bc))

                df.loc[len(df)] = [point_a, point_b, rad, euc, theta/math.pi,
                                   generate_ellipse(point_a, point_b), list(), list()]

    return df

def ellipse_dataframe(net, center, radius, nb_samples):
    graph = from_sumo_to_igraph_network(net)

    df = get_all_ellipses(center, radius, nb_samples)

    for node in graph.vs:
        e_id, direction = node['name'].split('_')
        e = net.getEdge(e_id)

        # Ending point of an edge
        to_n = e.getToNode()
        to_x, to_y = to_n.getCoord()
        to_lon, to_lat = net.convertXY2LonLat(to_x, to_y)
        to_p = shapely.Point(to_lon, to_lat)

        # Starting point of an edge
        from_n = e.getFromNode()
        from_x, from_y = from_n.getCoord()
        from_lon, from_lat = net.convertXY2LonLat(from_x, from_y)
        from_p = shapely.Point(from_lon, from_lat)

        for i in range(df.shape[0]):
            ellipse = df.at[i, 'ellipse']
            if shapely.contains(ellipse, to_p) and shapely.contains(ellipse, from_p):
                df.at[i, 'vs'] = df.at[i, 'vs'] + [node]
                df.at[i, 'es'] = df.at[i, 'vs'] + [e_id]

    df['subgraph'] = np.nan
    df['density'] = np.nan
    df['drivable_surface'] = np.nan

    for i in range(df.shape[0]):
        subgraph = graph.subgraph(df.at[i, 'vs'])
        df.at[i, 'subgraph'] = subgraph
        df.at[i, 'density'] = subgraph.density()

        # Computation of geodesic area
        geod = Geod(ellps="WGS84")
        area = abs(geod.geometry_area_perimeter(ellipse)[0])
        total_length = sum(list(map(lambda x: net.getEdge(x).getLength()/1000, df.at[i, 'es'])))
        df.at[i, 'drivable_surface'] = total_length / area

    return df

def ellipse_index(city_name, center, radius, nb_samples):
    net = sumolib.net.readNet('data/road_network/'+city_name+'_road_network.net.xml')

    df = ellipse_dataframe(net, center, radius, nb_samples)

    print("# of samples by radius")
    for r in df.radius.unique():
        print('%s: %s' % (r, df.loc[df['radius']==r].shape[0]))

    # We round all the data to avoid having to sparse data due to infinite decomals
    df['ratio'] = df.euclidean / df.radius
    df['ratio'] = df['ratio'].divide(5).round(decimals=2).mul(5)
    df['theta'] = df['theta'].divide(2).round(decimals=2).mul(2)
    df['euclidean'] = df['euclidean'].round(decimals=2)

    # We save the dataframe to csv in case we want to use it later
    df.to_csv('data/OD_samples/'+city_name+'_ellipse.csv', index=False)

    density_r(df, city_name)
    density_theta(df, city_name)
    density_d_r(df, city_name)
    density_d(df, city_name)

    drivable_surface_r(df, city_name)
    drivable_surface_theta(df, city_name)
    drivable_surface_d_r(df, city_name)
    drivable_surface_d(df, city_name)

    return df

def density_r(df, city):
    density = df.groupby('radius')['density'].mean().to_numpy()

    radius = df.radius.unique()

    plt.title('Density by radius in ' + city, size=16)
    plt.plot(radius, density, marker='o')
    plt.ylabel('density')
    plt.xlabel('r')

    plt.savefig('results/density/' + city + '_r.png')
    plt.show()

def density_theta(df, city):
    density = df.groupby(['radius', 'theta'])['density'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in density.radius.unique():
        density_rad = np.transpose(density.loc[density['radius']==rad][['theta', 'density']].to_numpy())
        x.append(density_rad[0])
        y.append(density_rad[1])

    plt.title('Density of elliptical subgraph by angle for each radius in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Density')
    plt.xlabel('theta')
    plt.legend(density.radius.unique())

    plt.savefig('results/density/' + city + '_theta.png')
    plt.show()

def density_d_r(df, city):
    density = df.groupby(['radius', 'ratio'])['density'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in density.radius.unique():
        density_rad = np.transpose(density.loc[density['radius']==rad][['ratio', 'density']].to_numpy())
        x.append(density_rad[0])
        y.append(density_rad[1])

    plt.title('Density of elliptical subgraph by ratio for each radius in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Density')
    plt.xlabel('d/r')
    plt.legend(density.radius.unique())

    plt.savefig('results/density/' + city + '_d_r.png')
    plt.show()

def density_d(df, city):
    density = df.groupby(['radius', 'euclidean'])['density'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in density.radius.unique():
        density_rad = np.transpose(density.loc[density['radius']==rad][['euclidean', 'density']].to_numpy())
        x.append(density_rad[0])
        y.append(density_rad[1])

    plt.title('Density of elliptical subgraph by euclidean distance in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Density')
    plt.xlabel('d')
    plt.legend(density.radius.unique())

    plt.savefig('results/density/' + city + '_d.png')
    plt.show()

def drivable_surface_r(df, city):
    drivable_surface = df.groupby('radius')['drivable_surface'].mean().to_numpy()

    radius = df.radius.unique()

    plt.title('Drivable surface by radius in ' + city, size=16)
    plt.plot(radius, drivable_surface, marker='o')
    plt.ylabel('Drivable surface')
    plt.xlabel('r')

    plt.savefig('results/drivable_surface/' + city + '_r.png')
    plt.show()

def drivable_surface_theta(df, city):
    drivable_surface = df.groupby(['radius', 'theta'])['drivable_surface'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in drivable_surface.radius.unique():
        drivable_surface_rad = np.transpose(drivable_surface.loc[drivable_surface['radius']==rad][['theta', 'drivable_surface']].to_numpy())
        x.append(drivable_surface_rad[0])
        y.append(drivable_surface_rad[1])

    plt.title('Drivable surface of elliptical subgraph by angle for each radius in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Drivable surface')
    plt.xlabel('theta')
    plt.legend(drivable_surface.radius.unique())

    plt.savefig('results/drivable_surface/' + city + '_theta.png')
    plt.show()

def drivable_surface_d_r(df, city):
    drivable_surface = df.groupby(['radius', 'ratio'])['drivable_surface'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in drivable_surface.radius.unique():
        drivable_surface_rad = np.transpose(drivable_surface.loc[drivable_surface['radius']==rad][['ratio', 'drivable_surface']].to_numpy())
        x.append(drivable_surface_rad[0])
        y.append(drivable_surface_rad[1])

    plt.title('Drivable surface of elliptical subgraph by ratio for each radius in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Drivable surface')
    plt.xlabel('d/r')
    plt.legend(drivable_surface.radius.unique())

    plt.savefig('results/drivable_surface/' + city + '_d_r.png')
    plt.show()

def drivable_surface_d(df, city):
    drivable_surface = df.groupby(['radius', 'euclidean'])['drivable_surface'].mean().reset_index(level=[0, 1])

    x, y = [], []

    for rad in drivable_surface.radius.unique():
        drivable_surface_rad = np.transpose(drivable_surface.loc[drivable_surface['radius']==rad][['euclidean', 'drivable_surface']].to_numpy())
        x.append(drivable_surface_rad[0])
        y.append(drivable_surface_rad[1])

    plt.title('Drivable surface of elliptical subgraph by euclidean distance in ' + city, size=16)
    for i in range(len(x)):
        plt.plot(x[i], y[i], marker='o')

    plt.ylabel('Drivable surface')
    plt.xlabel('d')
    plt.legend(drivable_surface.radius.unique())

    plt.savefig('results/drivable_surface/' + city + '_d.png')
    plt.show()
