from config import porto_config
from utils import setup_seed, get_traj_train_set, get_traj_tte_set, get_traj_srh_set
from road_network import RoadNetwork
from model import UniTR
from tasks import seg_type_cls, seg_speed_inf, traj_time_est, traj_sim_srh

import os
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    mode = args.mode

    # load configuration
    config = porto_config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    setup_seed(config['seed'])

    # load road network
    RN = RoadNetwork()
    RN.load('data/Porto.graphml')
    RN.process(shp_path='data/shp/')
    RN.extract_segment_attributes_and_labels()

    model = UniTR(params=config, road_network=RN).to(device)

    traj_path = 'data/daily_trajectory'
    traj_files = sorted([os.path.join(traj_path, f) for f in os.listdir(traj_path)])

    if mode == 'train':
        train_traj_df = get_traj_train_set(traj_files[:162], num_samples=config['num_train_trajs'], seed=config['seed'])
        model.train_process(train_traj_df)
    elif mode == 'test':
        chk = './checkpoints/UniTR_Porto.pth'
        model.load(chk, device)
        emb_path = f'checkpoints/segment_emb.npz'
        seg_type_cls.evaluation_with_emb(config['city_name'], emb_path, 'data/segment_type_label.npz',
                                         num_fold=100, num_classes=5, epochs=100, device=device)
        seg_speed_inf.evaluation_with_emb(config['city_name'], emb_path, 'data/segment_speed_label.npz',
                                          num_fold=5, epochs=100, device=device)
        tte_traj_df = get_traj_tte_set(traj_files[-31:])
        traj_time_est.evaluation(config['city_name'], tte_traj_df, model, epochs=100, device=device)

        query_traj_df, base_traj_df = get_traj_srh_set(traj_files[-31:], padding_idx=RN.num_seg_ids,
                                                       seed=config['seed'])
        traj_sim_srh.evaluation(config['city_name'], query_traj_df, base_traj_df, model, device=device)