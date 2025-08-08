#!/usr/bin/env python3
"""
span_segmentation_visualizer.py — (V6.4, 更换聚类算法)
====================================================================
用基于“连通性”的自定义聚类算法替换DBSCAN，以解决稀疏电塔的识别问题。

1.  **核心修改**:
    - 替换 `get_tower_centers` 函数的内部实现。新算法不再使用DBSCAN，
      而是通过广度优先搜索寻找点云中所有相互连通的部分，每个连通部分
      即为一个塔簇。
    - 这种方法不再依赖“局部密度”，只关心点是否在空间上“可达”，对
      稀疏点云更友好。
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import laspy
from laspy import LasHeader, LasData, ExtraBytesParams
import os
import colorsys
import sys # 新增：用于处理命令行参数
import json

CONFIG = {
    # =============================== 文件设置 ===============================
    # "infile": "processed_B线路已抽稀.las", # 移除写死的路径
    # "outfile": "00.las",                 # 移除写死的路径
    "label_attr": "classification",

    # =================================== 电塔检测 ===================================
    "tower_radius": 1.0,
    "tower_density": 40,
    "tower_hmin": 10.0,
    "tower_expand": 3.0,

    # =============================== 拓扑发现与跨段过滤 ===============================
    "tower_cluster_eps": 40.0,          # 新算法中，此参数定义了点与点之间多近才算“连通”
    "topology_k_neighbors": 3,
    "span_validation_min_points": 100,
    "tower_cluster_min_points": 3,      # 新算法中，此参数代表一个连通簇的最小总点数
    "protection_radius": 20.0,
    "span_width_buffer": 20.0,
    "span_box_height": 1000.0,

    # =================================== 去噪与过滤 ===================================
    "noise_nb": 10,
    "noise_r": 1.5,
}

# ------------------------------------------------------------------
# 辅助函数 (此部分无变化)
# ------------------------------------------------------------------
def load_cloud(path, label_attr=None):
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    original_classification = las.classification
    return xyz, original_classification

def remove_outliers(xyz, nb, r):
    if nb <= 0 or r <= 0 or len(xyz) < nb: return xyz
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, idx = pcd.remove_radius_outlier(nb_points=nb, radius=r)
    return xyz[idx] if len(idx) else xyz

def tower_mask(xyz, radius, density, hmin, expand):
    tree = cKDTree(xyz[:, :2])
    cnt = tree.query_ball_point(xyz[:, :2], radius, return_length=True)
    dense = cnt >= density
    zmax_list = [xyz[n][:, 2].max() if n and len(n) > 0 else xyz[i, 2] for i, n in enumerate(tree.query_ball_point(xyz[:, :2], radius))]
    zmax = np.array(zmax_list)
    tall = (zmax - xyz[:, 2]) >= hmin
    m = dense & tall
    if m.any():
        base = xyz[m][:, 2].min()
        xy_t = xyz[m][:, :2]
        dist = cKDTree(xy_t).query(xyz[:, :2], k=1, distance_upper_bound=expand)[0]
        m |= (dist < expand) & (xyz[:, 2] >= base)
    return m

def generate_distinct_colors(n):
    if n == 0:
        return np.array([])
    hues = np.linspace(0, 1, n, endpoint=False)
    colors_float = [colorsys.hls_to_rgb(h, 0.6, 0.8) for h in hues]
    colors_uint8 = (np.array(colors_float) * 255).astype(np.uint8)
    return colors_uint8

# ------------------------------------------------------------------
# 核心算法 (此部分无变化)
# ------------------------------------------------------------------
def get_tower_centers(xyz_t, cfg):
    min_total_points = cfg.get("tower_cluster_min_points", 3)
    if len(xyz_t) < min_total_points: return [], None
    tree = cKDTree(xyz_t)
    visited = np.zeros(len(xyz_t), dtype=bool)
    all_clusters = []
    for i in range(len(xyz_t)):
        if not visited[i]:
            new_cluster_indices = []; queue = [i]; visited[i] = True
            while queue:
                current_idx = queue.pop(0)
                new_cluster_indices.append(current_idx)
                neighbors = tree.query_ball_point(xyz_t[current_idx], r=cfg['tower_cluster_eps'])
                for neighbor_idx in neighbors:
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        queue.append(neighbor_idx)
            if len(new_cluster_indices) >= min_total_points:
                all_clusters.append(new_cluster_indices)
    centers = []; final_labels = np.full(len(xyz_t), -1, dtype=int)
    for i, cluster_indices in enumerate(all_clusters):
        tower_points = xyz_t[cluster_indices]
        min_bound = np.min(tower_points[:, :2], axis=0)
        max_bound = np.max(tower_points[:, :2], axis=0)
        center_xy = (min_bound + max_bound) / 2
        center_z = np.mean(tower_points[:, 2])
        centers.append([center_xy[0], center_xy[1], center_z])
        final_labels[cluster_indices] = i
    return np.array(centers), final_labels

def get_oriented_bbox_for_span(p_start, p_end, width, height):
    center = (p_start + p_end) / 2
    vec = p_end - p_start; dist = np.linalg.norm(vec)
    if dist < 1e-6: return None
    x_axis = vec / dist
    y_axis = np.cross(x_axis, np.array([0, 0, 1])) if abs(x_axis[2]) < 0.99 else np.cross(x_axis, np.array([0, 1, 0]))
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    rotation = np.array([x_axis, y_axis, z_axis]).T
    extent = np.array([dist, width, height])
    return o3d.geometry.OrientedBoundingBox(center, rotation, extent)

def discover_topology_and_spans(tower_centers, xyz_w, cfg):
    if len(tower_centers) < 2: return []
    wire_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_w))
    tower_tree = cKDTree(tower_centers)
    validated_spans = set()
    for i, p_start in enumerate(tower_centers):
        k = min(cfg['topology_k_neighbors'] + 1, len(tower_centers))
        _, neighbor_indices = tower_tree.query(p_start, k=k)
        for neighbor_idx in neighbor_indices:
            if i == neighbor_idx: continue
            p_end = tower_centers[neighbor_idx]
            span_tuple = tuple(sorted((i, neighbor_idx)))
            if span_tuple in validated_spans: continue
            bbox = get_oriented_bbox_for_span(p_start, p_end, cfg['span_width_buffer'], cfg['span_box_height'])
            if bbox is not None and len(bbox.get_point_indices_within_bounding_box(wire_pcd.points)) >= cfg['span_validation_min_points']:
                validated_spans.add(span_tuple)
    return list(validated_spans)

# ------------------------------------------------------------------
# 主执行函数
# ------------------------------------------------------------------
def main(input_path, output_path):
    cfg = CONFIG

    print(f"正在加载文件: {input_path}")
    print("模式: 跨段分割可视化")

    # 加载所有点和原始分类信息
    xyz_all, original_classification = load_cloud(input_path)
    if len(xyz_all) == 0:
        print("错误：文件中没有找到任何点云数据。"); return

    # 分离地面点和非地面点
    ground_mask = original_classification == 0
    non_ground_mask = original_classification == 1
    xyz_ground = xyz_all[ground_mask]
    xyz_non_ground = xyz_all[non_ground_mask]

    print(f"载入 {len(xyz_ground)} 个地面点，{len(xyz_non_ground)} 个非地面点")

    print("1. 正在进行电塔识别...")
    # 只对非地面点进行电塔识别
    m_tower = tower_mask(xyz_non_ground,
                         radius=cfg["tower_radius"],
                         density=cfg["tower_density"],
                         hmin=cfg["tower_hmin"],
                         expand=cfg["tower_expand"])
    xyz_t, xyz_w = xyz_non_ground[m_tower], xyz_non_ground[~m_tower]
    print(f" -> 识别出 {len(xyz_t)} 个电塔点，{len(xyz_w)} 个潜在导线/非塔点。")
    if len(xyz_t) == 0: print("错误：未识别出任何电塔点。"); return

    print(" -> 对非塔点进行初步去噪...")
    xyz_w = remove_outliers(xyz_w, cfg["noise_nb"], cfg["noise_r"])

    print("2. 正在分割独立电塔...")
    tower_centers, tower_segmentation_labels = get_tower_centers(xyz_t, cfg)
    print(f" -> 找到并定位了 {len(tower_centers)} 座独立电塔。")
    if len(tower_centers) < 2: print("电塔数量不足2，无法继续。"); return
    
    # 根据电塔的聚类结果生成电塔的json文件以便展示
    print(" -> 正在生成 tower.json...")
    tower_info_list = []
    # 确保 tower_segmentation_labels 不是 None 并且有内容
    if tower_segmentation_labels is not None and len(tower_segmentation_labels) > 0:
        for i in range(len(tower_centers)):
            # 筛选出当前塔簇的所有点
            cluster_points = xyz_t[tower_segmentation_labels == i]
            if len(cluster_points) > 0:
                tower_height = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
                point_count = len(cluster_points)
                
                tower_info_list.append({
                    "tower_id": i + 1,
                    "point_count": int(point_count),
                    "height_m": float(f"{tower_height:.2f}"),
                    "center_x": float(f"{tower_centers[i][0]:.2f}"),
                    "center_y": float(f"{tower_centers[i][1]:.2f}"),
                    "center_z": float(f"{tower_centers[i][2]:.2f}")
                })

    # 定义 tower.json 的输出路径，并保存文件
    tower_json_path = os.path.join(os.path.dirname(output_path), 'tower.json')
    with open(tower_json_path, 'w', encoding='utf-8') as f:
        json.dump(tower_info_list, f, indent=4, ensure_ascii=False)
    print(f" -> 已保存电塔信息到: {tower_json_path}")


    print("3. 正在通过导线数据发现电塔拓扑...")
    spans = discover_topology_and_spans(tower_centers, xyz_w, cfg)
    print(f" -> 发现了 {len(spans)} 个有效的物理跨段。")
    if not spans: print("未能发现任何有效跨段。"); return

    num_spans = len(spans)
    palette = generate_distinct_colors(num_spans)
    print(f" -> 已为 {num_spans} 个潜在跨段生成了独特的颜色。")

    print("4. 正在为每个跨段分配ID并进行着色...")
    span_labels = np.zeros(len(xyz_w), dtype=int)
    wire_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_w))

    for i, (t_idx1, t_idx2) in enumerate(spans):
        span_id = i + 1
        center_A, center_B = tower_centers[t_idx1], tower_centers[t_idx2]
        vec = center_B - center_A
        vec_len = np.linalg.norm(vec)
        if vec_len < cfg['protection_radius'] * 2: continue

        direction = vec / vec_len
        p_start = center_A + direction * cfg['protection_radius']
        p_end = center_B - direction * cfg['protection_radius']

        bbox = get_oriented_bbox_for_span(p_start, p_end, cfg['span_width_buffer'], cfg['span_box_height'])
        if bbox is None: continue

        span_indices = bbox.get_point_indices_within_bounding_box(wire_pcd.points)
        if not span_indices: continue

        mask = span_labels[span_indices] == 0
        updatable_indices = np.array(span_indices)[mask]
        span_labels[updatable_indices] = span_id

    print("5. 正在保存可视化结果...")

    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(ExtraBytesParams(name="span_id", type="uint16"))
    hdr.add_extra_dim(ExtraBytesParams(name="tower_id", type="uint16"))
    hdr.add_extra_dim(ExtraBytesParams(name="process", type="uint16"))
    out = laspy.LasData(hdr)

    all_xyz = np.vstack([xyz_t, xyz_w, xyz_ground])
    out.xyz = all_xyz

    rgb_w = np.full((len(xyz_w), 3), [255, 255, 255], dtype=np.uint8)
    valid_mask = span_labels > 0
    if num_spans > 0 and np.any(valid_mask):
        rgb_w[valid_mask] = palette[span_labels[valid_mask] - 1]

    tower_color = np.array([173, 216, 230], dtype=np.uint8)
    rgb_t = np.tile(tower_color, (len(xyz_t), 1))

    ground_color = np.array([128, 128, 128], dtype=np.uint8)
    rgb_ground = np.tile(ground_color, (len(xyz_ground), 1))

    rgb_all = np.vstack([rgb_t, rgb_w, rgb_ground])
    out.red = (rgb_all[:, 0].astype(np.uint16) << 8)
    out.green = (rgb_all[:, 1].astype(np.uint16) << 8)
    out.blue = (rgb_all[:, 2].astype(np.uint16) << 8)

    out.classification = np.hstack([np.full(len(xyz_t), 2, np.uint8),
                                     np.full(len(xyz_w), 1, np.uint8),
                                     np.full(len(xyz_ground), 0, np.uint8)])

    out.span_id = np.hstack([np.zeros(len(xyz_t), np.uint16),
                             span_labels.astype(np.uint16),
                             np.zeros(len(xyz_ground), np.uint16)])

    tower_id_t = np.zeros(len(xyz_t), dtype=np.uint16)
    if tower_segmentation_labels is not None:
        valid_mask_t = tower_segmentation_labels != -1
        tower_id_t[valid_mask_t] = (tower_segmentation_labels[valid_mask_t] + 1).astype(np.uint16)

    out.tower_id = np.hstack([tower_id_t,
                              np.zeros(len(xyz_w), np.uint16),
                              np.zeros(len(xyz_ground), np.uint16)])

    process_w = np.ones(len(xyz_w), dtype=np.uint16)
    unique_span_ids = np.unique(span_labels[span_labels > 0])

    for i, span_id in enumerate(unique_span_ids):
        span_mask = span_labels == span_id
        process_w[span_mask] = i + 2

    out.process = np.hstack([np.full(len(xyz_t), 1, np.uint16),
                             process_w,
                             np.zeros(len(xyz_ground), np.uint16)])

    out.write(output_path) # 使用 output_path 参数
    final_span_count = len(np.unique(span_labels[span_labels > 0]))
    print(f"* 保存 {output_path} — 共可视化 {final_span_count} 个跨段，包含 {len(xyz_ground)} 个地面点。")
    print(f"  Process值分配：地面=0，电塔=1，跨段=2-{1+final_span_count}")

# ------------------------------------------------------------------
# 脚本入口：解析命令行参数
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python fit.py <input_file_path> <output_file_path>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"错误: 输入文件不存在 -> {in_path}")
        sys.exit(1)

    main(in_path, out_path)