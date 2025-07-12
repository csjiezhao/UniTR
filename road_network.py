import osmnx as ox
import networkx as nx
import numpy as np
import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import geojson
from tqdm import tqdm
from geopy.distance import geodesic
import matplotlib.pyplot as plt

ox.settings.use_cache = True
ox.settings.log_console = True


class RoadNetwork:
    """
    Some important fields
    self.G, self.G_edge2idx, self.G_idx2edge, self.G_node_gdf, self.G_edge_gdf
    self.LG, self.LG_adj_sp, self.LG_adj
    self.segment_id_enc, self.segment_len_enc, self.segment_lng_enc, self.segment_lat_enc
    self.segment_type_label, self.segment_oneway_label, self.segment_bridge_label, self.segment_tunnel_label
    self.segment_speed_label
    """
    def __init__(self, city_name=None, boundary=None, save_path=None,
                 network_type='drive', simplify=True, truncate_by_edge=True):
        if (city_name is not None) and (boundary is not None):
            raise ValueError('Only one download mode should be specified! (by city or by boundary)')

        if city_name is not None:
            self.G = ox.graph_from_place(city_name, network_type=network_type,
                                         simplify=simplify, truncate_by_edge=truncate_by_edge)
            self.G.graph['name'] = city_name
            self.save(save_path, city_name)
            print(self.G)

        if boundary is not None:
            n, s, e, w = boundary
            self.G = ox.graph_from_bbox(north=n, south=s, east=e, west=w,
                                        simplify=simplify, truncate_by_edge=truncate_by_edge)
            self.save(save_path, str(boundary))
            print(self.G)

    def save(self, path, name):
        ox.save_graphml(self.G, os.path.join(path, f'{name}.graphml'))

    def load(self, file):
        self.G = ox.load_graphml(file)

    def process(self, shp_path):
        """
        Processing road network data
        """
        '''1.create index for each edge'''
        edge_list = list(self.G.edges)
        edge2idx = {e: i for i, e in enumerate(edge_list)}
        idx2edge = {i: e for i, e in enumerate(edge_list)}
        nx.set_edge_attributes(self.G, values=edge2idx, name='edge_idx')  # set edge index to graph edge attribute
        self.G_edge2idx = edge2idx
        self.G_idx2edge = idx2edge

        '''2.get node and edge information'''
        self.G_node_gdf, self.G_edge_gdf = ox.graph_to_gdfs(self.G)
        self.G_node_gdf.reset_index(inplace=True)
        self.G_edge_gdf.reset_index(inplace=True)
        midpoints = self.G_edge_gdf['geometry'].interpolate(0.5, normalized=True)
        midpoints = midpoints.map(lambda x: x.coords[0])
        self.G_edge_gdf['midpoint'] = midpoints
        uvk = self.G_edge_gdf[['u', 'v', 'key']].apply(tuple, axis=1)
        edge2midpoint = dict(zip(uvk, midpoints))
        nx.set_edge_attributes(self.G, values=edge2midpoint, name='midpoint')

        '''3.save Shapefile for map matching'''
        self.G_node_gdf.to_file(os.path.join(shp_path, 'G_nodes.shp'),
                                driver='ESRI Shapefile', encoding='utf-8')
        self.G_edge_gdf[['u', 'v', 'key', 'edge_idx', 'geometry']].to_file(os.path.join(shp_path, 'G_edges.shp'),
                                                                           driver='ESRI Shapefile', encoding='utf-8')

        '''4.generate line graph (LG)'''
        LG = nx.line_graph(self.G)
        # note: the node attrs of LG should be obtained from the edge attrs of G
        LG_node_attrs = {e: self.G.get_edge_data(*e) for e in self.G.edges}
        nx.set_node_attributes(LG, LG_node_attrs)  # update node attrs of LG
        LG = nx.relabel.relabel_nodes(LG, edge2idx)  # use edge indices to relabel nodes of LG
        # resort nodes of LG according to edge index
        sorted_LG = nx.MultiDiGraph()
        sorted_LG.add_nodes_from(sorted(LG.nodes(data=True)))
        sorted_LG.add_edges_from(LG.edges(data=True))
        self.LG = sorted_LG

        '''5.get adjacency matrix of LG'''
        self.LG_adj_sp = nx.adjacency_matrix(self.LG)
        self.LG_adj = self.LG_adj_sp.todense()

    def extract_segment_attributes_and_labels(self, save_path=None, len_bin_size=50, coord_bin_size=0.001):
        """
        :param len_bin_size:
        :param coord_bin_size:
        :return:
        """
        '''extract segment attribute: id'''
        self.segment_id_enc = self.G_edge_gdf['edge_idx'].values
        self.num_seg_ids = len(self.segment_id_enc)

        '''extract segment attribute: length'''
        self.segment_len_enc = self.G_edge_gdf['length'].map(lambda x: math.floor(x / len_bin_size)).values
        self.num_seg_len_bins = max(self.segment_len_enc) + 1

        '''extract segment attribute: coordinates in midpoint'''
        lng0, lat0, lng1, lat1 = self.G_edge_bounds
        self.num_seg_lng_bins = int(np.ceil((lng1 - lng0) / coord_bin_size))
        lng_bins = [lng0 + i * coord_bin_size for i in range(self.num_seg_lng_bins + 1)]
        self.num_seg_lat_bins = int(np.ceil((lat1 - lat0) / coord_bin_size))
        lat_bins = [lat0 + i * coord_bin_size for i in range(self.num_seg_lat_bins + 1)]

        mid_lng = self.G_edge_gdf['midpoint'].map(lambda x: x[0])
        mid_lat = self.G_edge_gdf['midpoint'].map(lambda x: x[1])
        self.segment_lng_enc = np.asarray(pd.cut(mid_lng, bins=lng_bins, labels=list(np.arange(self.num_seg_lng_bins))))
        self.segment_lat_enc = np.asarray(pd.cut(mid_lat, bins=lat_bins, labels=list(np.arange(self.num_seg_lat_bins))))

        self.segment_attr_enc = np.column_stack((self.segment_id_enc, self.segment_len_enc,
                                                 self.segment_lng_enc, self.segment_lat_enc))

        '''extract segment visual feature'''
        self.segment_vis_feat = np.load('data/segment_vis_feat.npz')['data']

        '''extract segment label: highway (aka road type)'''
        highway = self.G_edge_gdf['highway'].map(lambda x: x[0] if type(x) == list else x)
        highway2idx = {
            'primary': 0,
            'primary_link': 0,
            'secondary': 1,
            'secondary_link': 1,
            'tertiary': 2,
            'tertiary_link': 2,
            'residential': 3,
            'living_street': 3,
        }
        highway = highway.map(lambda x: highway2idx[x] if x in highway2idx else -1)
        self.segment_type_label = highway.values

        '''extract segment label: oneway'''
        oneway = self.G_edge_gdf['oneway'].map(lambda x: 1 if x is True else 0)
        self.segment_oneway_label = oneway.values

        '''extract segment label: bridge'''
        bridge = self.G_edge_gdf['bridge'].fillna(0)
        bridge.loc[bridge != 0] = 1
        self.segment_bridge_label = bridge.values

        '''extract segment label: tunnel'''
        tunnel = self.G_edge_gdf['tunnel'].fillna(0)
        tunnel.loc[tunnel != 0] = 1
        self.segment_tunnel_label = tunnel.values

        if save_path:
            print(self.segment_type_label, [e for e in self.segment_type_label if e == 4])
            np.savez_compressed(os.path.join(save_path, 'segment_type_label.npz'), data=self.segment_type_label)
            np.savez_compressed(os.path.join(save_path, 'segment_oneway_label.npz'), data=self.segment_oneway_label)
            np.savez_compressed(os.path.join(save_path, 'segment_bridge_label.npz'), data=self.segment_bridge_label)
            np.savez_compressed(os.path.join(save_path, 'segment_tunnel_label.npz'), data=self.segment_tunnel_label)

    def segment_attribute_statistics(self):
        for col in self.G_edge_gdf.columns:
            pct_missing = np.mean(self.G_edge_gdf[col].isnull())
            print('{} - {}%'.format(col, round(pct_missing * 100)))

    @property
    def G_node_bounds(self):
        return self.G_node_gdf.geometry.total_bounds

    @property
    def G_edge_bounds(self):
        return self.G_edge_gdf.geometry.total_bounds

    def get_bound_polygon(self, result_type='Polygon'):
        lng0, lat0, lng1, lat1 = self.G_edge_bounds
        ns_dist = geodesic((lat0, lng0), (lat1, lng0)).m
        we_dist = geodesic((lat0, lng0), (lat0, lng1)).m
        print('NS Dist(height):', ns_dist, 'WE Dist(width):', we_dist)
        bound_polygon = Polygon([(lng0, lat0), (lng0, lat1), (lng1, lat1), (lng1, lat0), (lng0, lat0)])
        if result_type == 'Polygon':
            return bound_polygon
        elif result_type == 'GeoJson':
            feature_collection = geojson.FeatureCollection([])
            feature = geojson.Feature(
                geometry=bound_polygon,
            )
            feature_collection.features.append(feature)
            geo_str = geojson.dumps(feature_collection, sort_keys=True)
            return geo_str
        elif result_type == 'GeoSeries':
            return gpd.GeoSeries(bound_polygon)
        else:
            raise TypeError('Unsupported result_type for bound_polygon')

    def draw_graph(self, save_path):
        fig, ax = plt.subplots(1, 1, figsize=(36, 16))

        G_pos = {}
        for node_id, node_data in self.G.nodes(data=True):
            G_pos[node_id] = np.array([node_data['x'], node_data['y']])
        nx.draw_networkx_nodes(self.G, G_pos, node_size=25, node_color='#999999', node_shape='.', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(self.G, G_pos, width=3, edge_color='#999999', alpha=0.7, arrows=False, ax=ax)

        label_values = [0, 1, 2, 3, 4, -1]
        label_names = ['primary', 'secondary', 'tertiary', 'residential', 'living street', 'others']
        colors = ['red', 'orange', 'yellow', 'skyblue', 'lime', 'purple']

        for c, label, label_name in zip(colors, label_values, label_names):
            temp_nodes = [i for i, n in enumerate(self.segment_type_label) if n == label]
            LG_sub = self.LG.subgraph(temp_nodes)
            LG_sub_pos = {k: v['midpoint'] for k, v in LG_sub.nodes(data=True)}
            nx.draw_networkx_nodes(LG_sub, LG_sub_pos, node_size=15, node_color=c, ax=ax, label=label_name)
        plt.legend(markerscale=4, fontsize=30)
        plt.savefig(os.path.join(save_path, 'road_network_vis.pdf'))
        fig.show()

    def compute_segment_speed_label(self, tra_file, save_path):
        tra_df = pd.read_csv(tra_file, index_col=None, usecols=['cpath', 'pass_time'])
        segment_speed_dict = {}
        for idx, row in tqdm(tra_df.iterrows()):
            cpath, pass_time = eval(row['cpath']), eval(row['pass_time'])
            for i, seg in enumerate(cpath):
                # compute segment speed
                if pass_time[i] > 0:
                    speed_temp = self.LG.nodes[seg]['length'] / pass_time[i]
                    if speed_temp < 30:
                        avg, n = segment_speed_dict.get(seg, (0, 0))
                        segment_speed_dict[seg] = ((avg * n + speed_temp) / (n + 1), (n + 1))

        self.segment_speed_label = np.array([segment_speed_dict.get(i, (0, 0))[0]
                                             for i in range(self.LG.number_of_nodes())])
        if save_path:
            np.savez_compressed(os.path.join(save_path, 'segment_speed_label.npz'), data=self.segment_speed_label)


if __name__ == '__main__':

    RN = RoadNetwork()
    RN.load('data/Porto.graphml')
    RN.process(shp_path='data/shp/')
    RN.extract_segment_attributes_and_labels(save_path='data/')
    RN.G_edge_gdf[['edge_idx', 'geometry']].to_csv('data/segment_geometry.csv', index=False)



