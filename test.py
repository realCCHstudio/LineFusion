#!/usr/bin/env python3
"""
wire_extraction.py — (V5, 逻辑修正) 迭代式中心发散生长 + (V7, 量化分析)
====================================================================
v_Pinnacle_v5 (iterative-center-out-v2): 根据用户反馈修正迭代逻辑。

1.  **宏观分割**: 策略不变。
2.  **微观生长**: 采用修正后的“迭代式中心发散”策略：
    - **详尽提取**: 在生长阶段，不再提前过滤小簇。算法会尽其所能，
      将跨段内所有能识别的线状物（无论大小）全部提取并标记。
    - **统一过滤**: 将噪声过滤步骤完全交还给 main 函数的后处理阶段，
      在全局视角下统一清除所有不满足条件的簇。
3.  **新增功能**: 对最终提取的导线进行量化分析，并输出linedata.json。
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import laspy
from laspy import LasHeader, LasData, ExtraBytesParams
import os
import json # <--- 新增导入
import sys # <--- 新增导入，用于确定路径

# --- 新增：确定资源路径 ---
# 在 Electron 打包后，资源文件位于 process.resourcesPath
# 这个函数帮助我们无论在开发环境还是打包后都能找到文件
def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        # 如果是打包后的应用 (pyinstaller or similar)
        base_path = sys._MEIPASS
    else:
        # 如果是正常的 python 环境
        # 在 Electron 中，我们从脚本的父目录寻找资源
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

CONFIG = {
    # ==================== 策略选择 ====================
    "growth_strategy": "iterative_center_out_v2", # 可选: "ends_in", "center_out", "iterative_center_out_v2"
    # ================================================

    # --- 修改：使用 get_resource_path 来定位输入文件 ---
    "infile": get_resource_path("processed_B线路已抽稀.las"),
    "outfile": "all_colored_output_final_v5.las",
    "label_attr": "classification",

    # -------- 电塔检测 --------
    "tower_radius": 1.0,
    "tower_density": 40,
    "tower_hmin": 10.0,
    "tower_expand": 3.0,

    # -------- 拓扑发现与跨段过滤 --------
    "tower_cluster_eps": 50.0,
    "topology_k_neighbors": 3,
    "span_validation_min_points": 100,
    "protection_radius": 5.0,
    "span_width_buffer": 10.0,

    # -------- 跨段内部智能生长参数 --------
    "seed_slice_width": 20.0,
    "seed_cluster_eps": 5.5,
    "merge_dist_thresh": 15.0,
    "center_seed_radius": 8.0,

    "path_lookback": 20,
    "perp_tolerance_radius": 4.0,
    "base_search_dist": 30.0,
    "max_search_dist": 70.0,
    "step_size": 8.0,

    # -------- 去噪与过滤 --------
    "noise_nb": 10,
    "noise_r": 1.5,
    "min_cluster_points": 50,
}

# ------------------------------------------------------------------
# Helpers (无变化)
# ------------------------------------------------------------------
def load_cloud(path, label_attr=None):
    # --- 新增：检查文件是否存在 ---
    if not os.path.exists(path):
        print(f"Error: Input file not found at {path}")
        sys.exit(1)
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    if label_attr:
        lbl = getattr(las, label_attr, None)
        mask = lbl == 1 if lbl is not None else las.classification == 1
    else: mask = las.classification == 1
    return xyz[mask]

def remove_outliers(xyz, nb, r):
    if nb <= 0 or r <= 0 or len(xyz) < nb: return xyz
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, idx = pcd.remove_radius_outlier(nb_points=nb, radius=r)
    return xyz[idx] if len(idx) else xyz

def tower_mask(xyz, radius, density, hmin, expand):
    tree = cKDTree(xyz[:, :2])
    cnt = tree.query_ball_point(xyz[:, :2], radius, return_length=True)
    dense = cnt >= density
    zmax_list = []
    for i, n in enumerate(tree.query_ball_point(xyz[:, :2], radius)):
        if n and len(n) > 0: zmax_list.append(xyz[n][:, 2].max())
        else: zmax_list.append(xyz[i, 2])
    zmax = np.array(zmax_list)
    tall = (zmax - xyz[:, 2]) >= hmin
    m = dense & tall
    if m.any():
        base = xyz[m][:, 2].min()
        xy_t = xyz[m][:, :2]
        dist = cKDTree(xy_t).query(xyz[:, :2], k=1, distance_upper_bound=expand)[0]
        m |= (dist < expand) & (xyz[:, 2] >= base)
    return m

def get_local_direction(points):
    if len(points) < 2: return None
    center = np.mean(points, axis=0)
    cov = np.cov((points - center).T)
    _, eigenvectors = np.linalg.eigh(cov)
    return eigenvectors[:, -1]

# ------------------------------------------------------------------
# 新增: 量化分析函数 (无变化)
# ------------------------------------------------------------------
def calculate_length(points):
    """计算点云簇的沿线长度"""
    if len(points) < 2: return 0.0
    main_dir = get_local_direction(points)
    if main_dir is None: return 0.0
    proj = points @ main_dir
    sorted_indices = np.argsort(proj)
    sorted_points = points[sorted_indices]
    distances = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    return np.sum(distances)

def calculate_width(points):
    """估算点云簇的平均直径"""
    if len(points) < 3: return 0.0
    main_dir = get_local_direction(points)
    if main_dir is None: return 0.0
    centroid = np.mean(points, axis=0)
    vecs_from_centroid = points - centroid
    proj_lengths = np.dot(vecs_from_centroid, main_dir)
    proj_vecs = proj_lengths[:, np.newaxis] * main_dir
    perp_vecs = vecs_from_centroid - proj_vecs
    perp_dists = np.linalg.norm(perp_vecs, axis=1)
    return np.mean(perp_dists) * 2

def calculate_curvature(points):
    """计算线缆的弧垂深度和弯曲度(弧垂比)"""
    if len(points) < 3: return 0.0, 0.0
    main_dir = get_local_direction(points)
    if main_dir is None: return 0.0, 0.0
    proj = points @ main_dir
    p_start, p_end = points[np.argmin(proj)], points[np.argmax(proj)]
    line_vec = p_end - p_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6: return 0.0, 0.0
    line_unit_vec = line_vec / line_len
    vecs_from_start = points - p_start
    cross_prods = np.cross(vecs_from_start, line_unit_vec)
    dists_to_line = np.linalg.norm(cross_prods, axis=1)
    sag_depth = np.max(dists_to_line)
    curvature_ratio = sag_depth / line_len if line_len > 0 else 0.0
    return sag_depth, curvature_ratio


# ------------------------------------------------------------------
# 核心算法 (无变化)
# ------------------------------------------------------------------
def get_tower_centers(xyz_t, cfg):
    if len(xyz_t) < 10: return [], None
    xyz_t_for_clustering = xyz_t.copy(); xyz_t_for_clustering[:, 2] = 0
    pcd_towers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_t_for_clustering))
    labels = np.array(pcd_towers.cluster_dbscan(eps=cfg['tower_cluster_eps'], min_points=10))
    centers = [np.mean(xyz_t[labels == l], axis=0) for l in np.unique(labels[labels != -1])]
    return np.array(centers), labels

def get_oriented_bbox_for_span(p_start, p_end, width, height=50):
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
            bbox = get_oriented_bbox_for_span(p_start, p_end, cfg['span_width_buffer'])
            if bbox is not None and len(bbox.get_point_indices_within_bounding_box(wire_pcd.points)) >= cfg['span_validation_min_points']:
                validated_spans.add(span_tuple)
    return list(validated_spans)

def grow_path(path_indices, local_labels, current_label, span_xyz, span_tree, main_dir, proj_min, proj_max, cfg, center_out_mode=False, available_indices=None):
    """(子函数) 智能生长核心逻辑"""
    span_length = proj_max - proj_min
    while True:
        front_points = span_xyz[path_indices[-(cfg['path_lookback']):]]
        if len(front_points) < 2: break
        direction = get_local_direction(front_points)
        if direction is None: break
        if np.dot(direction, main_dir) < 0:
            direction *= -1
        current_pos_proj = np.mean(front_points, axis=0) @ main_dir
        normalized_pos = (current_pos_proj - proj_min) / span_length if span_length > 0 else 0.5
        if center_out_mode:
            growth_factor = (2 * normalized_pos - 1)**2 if 0 <= normalized_pos <= 1 else 1
        else:
            growth_factor = 1 - (2 * normalized_pos - 1)**2 if 0 <= normalized_pos <= 1 else 0
        current_search_dist = cfg['base_search_dist'] + (cfg['max_search_dist'] - cfg['base_search_dist']) * growth_factor
        new_front_points = []
        front_center = np.mean(front_points, axis=0)
        candidate_indices = span_tree.query_ball_point(front_center, r=current_search_dist + cfg['step_size'])
        for cand_idx in candidate_indices:
            is_available = available_indices is None or cand_idx in available_indices
            if is_available and local_labels[cand_idx] == -1:
                vec_to_cand = span_xyz[cand_idx] - front_center
                if np.dot(vec_to_cand, direction) > 0:
                    perp_dist = np.linalg.norm(np.cross(vec_to_cand, direction))
                    if perp_dist < cfg['perp_tolerance_radius']:
                        new_front_points.append(cand_idx)
                        local_labels[cand_idx] = current_label
        if not new_front_points: break
        path_indices.extend(new_front_points)

def intelligent_wire_growth_iterative_center_out_v2(span_xyz, span_indices_original, all_labels, max_label, cfg):
    """【V5 策略】“迭代式中心向外”发散生长算法 (逻辑修正)"""
    if len(span_xyz) < 20: return max_label

    local_labels = np.full(len(span_xyz), -1, dtype=int)
    span_tree = cKDTree(span_xyz)
    main_dir = get_local_direction(span_xyz)
    if main_dir is None: return max_label

    remaining_indices = set(range(len(span_xyz)))

    iter_count = 0
    while True:
        if not remaining_indices:
            break

        iter_count += 1

        rem_idx_list = list(remaining_indices)
        rem_xyz = span_xyz[rem_idx_list]

        if len(rem_xyz) < 2: break

        lowest_rem_idx_in_rem = np.argmin(rem_xyz[:, 2])
        lowest_point_idx = rem_idx_list[lowest_rem_idx_in_rem]
        lowest_point_pos = span_xyz[lowest_point_idx]

        candidate_seeds = span_tree.query_ball_point(lowest_point_pos, r=cfg['center_seed_radius'])
        center_seed_indices = [idx for idx in candidate_seeds if idx in remaining_indices]

        if len(center_seed_indices) < 2:
            remaining_indices.remove(lowest_point_idx)
            continue

        current_wire_label = max_label
        local_labels[center_seed_indices] = current_wire_label

        path_positive, path_negative = list(center_seed_indices), list(center_seed_indices)
        proj = span_xyz @ main_dir
        proj_min, proj_max = np.min(proj), np.max(proj)

        grow_path(path_positive, local_labels, current_wire_label, span_xyz, span_tree, main_dir, proj_min, proj_max, cfg, center_out_mode=True, available_indices=remaining_indices)
        grow_path(path_negative, local_labels, current_wire_label, span_xyz, span_tree, -main_dir, -proj_max, -proj_min, cfg, center_out_mode=True, available_indices=remaining_indices)

        newly_labeled_mask = local_labels == current_wire_label
        newly_labeled_indices = set(np.where(newly_labeled_mask)[0])

        remaining_indices.difference_update(newly_labeled_indices)
        max_label += 1

    np.put(all_labels, span_indices_original, local_labels)
    return max_label

def main():
    cfg = CONFIG
    strategy = cfg['growth_strategy']
    cfg['outfile'] = f"output_strategy_{strategy}.las"

    print(f"Loading file: {cfg['infile']}")
    print(f"Using growth strategy: {strategy}")

    xyz_all = load_cloud(cfg["infile"])
    if len(xyz_all) == 0:
        print("Error: No point cloud data found in the file."); return

    print("1. Identifying towers...")
    m_tower = tower_mask(xyz_all, cfg["tower_radius"], cfg["tower_density"], cfg["tower_hmin"], cfg["tower_expand"])
    xyz_t, xyz_w = xyz_all[m_tower], xyz_all[~m_tower]
    print(f" -> Identified {len(xyz_t)} tower points and {len(xyz_w)} potential wire points.")
    if len(xyz_t) == 0: print("Error: No tower points identified."); return

    print(" -> Denoising wire points...")
    xyz_w = remove_outliers(xyz_w, cfg["noise_nb"], cfg["noise_r"])

    print("2. Segmenting individual towers...")
    tower_centers, tower_segmentation_labels = get_tower_centers(xyz_t, cfg)
    print(f" -> Found and located {len(tower_centers)} individual towers.")
    if len(tower_centers) < 2: print("Fewer than 2 towers found."); return

    print("3. Discovering topology from wire data...")
    spans = discover_topology_and_spans(tower_centers, xyz_w, cfg)
    print(f" -> Discovered {len(spans)} valid physical spans.")
    if not spans: print("Could not discover any valid spans."); return

    print("4. Performing intelligent growth within spans...")
    all_labels = np.full(len(xyz_w), -1, dtype=int)
    max_label_global = 0
    wire_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz_w))

    for i, (t_idx1, t_idx2) in enumerate(spans):
        print(f"Processing span {i+1}/{len(spans)} (Tower {t_idx1} <-> Tower {t_idx2})...")
        center_A, center_B = tower_centers[t_idx1], tower_centers[t_idx2]
        vec = center_B - center_A
        vec_len = np.linalg.norm(vec)
        if vec_len < cfg['protection_radius'] * 2: continue
        direction = vec / vec_len
        p_start = center_A + direction * cfg['protection_radius']
        p_end = center_B - direction * cfg['protection_radius']

        bbox = get_oriented_bbox_for_span(p_start, p_end, cfg['span_width_buffer'])
        if bbox is None: continue

        span_indices = bbox.get_point_indices_within_bounding_box(wire_pcd.points)
        if not span_indices: continue

        span_xyz = xyz_w[span_indices]

        if strategy == 'iterative_center_out_v2':
            max_label_global = intelligent_wire_growth_iterative_center_out_v2(span_xyz, span_indices, all_labels, max_label_global, cfg)
        else:
            print(f"Warning: Strategy '{strategy}' is not the expected V5 strategy.")
            pass

    print("5. Post-processing and saving...")
    label = all_labels
    unique_labels, counts = np.unique(label[label != -1], return_counts=True)
    small_clusters = unique_labels[counts < cfg['min_cluster_points']]
    if len(small_clusters) > 0:
        print(f" -> (Global Filter) Removing {len(small_clusters)} small wire clusters (points < {cfg['min_cluster_points']}).")
        for small_id in small_clusters:
            label[label == small_id] = -1

    unique_labels = np.unique(label[label != -1])
    map_ids = {old: new for new, old in enumerate(unique_labels)}
    label = np.array([map_ids.get(l, -1) for l in label])

    # --- Quantitative analysis and JSON output ---
    line_data_list = []
    final_labels = np.unique(label[label != -1])
    print(f"6. Calculating physical properties for {len(final_labels)} final wire clusters...")
    for line_id in final_labels:
        line_points = xyz_w[label == line_id]
        if len(line_points) < 10: continue

        length = calculate_length(line_points)
        width = calculate_width(line_points)
        sag_depth, curvature_ratio = calculate_curvature(line_points)

        line_info = {
            "line_name": f"Line_{line_id + 1}",
            "line_id": int(line_id + 1),
            "point_count": len(line_points),
            "estimated_length_m": round(length, 2),
            "estimated_diameter_m": round(width, 3),
            "sag_depth_m": round(sag_depth, 2),
            "curvature_ratio": round(curvature_ratio, 4)
        }
        line_data_list.append(line_info)

    json_outfile = "linedata.json"
    with open(json_outfile, 'w', encoding='utf-8') as f:
        json.dump(line_data_list, f, indent=4, ensure_ascii=False)
    print(f"✓ Line properties saved to {json_outfile}")
    # --- End of new feature ---

    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(ExtraBytesParams(name="line_id", type="uint16"))
    hdr.add_extra_dim(ExtraBytesParams(name="tower_id", type="uint16"))
    out = laspy.LasData(hdr)

    all_xyz = np.vstack([xyz_t, xyz_w])
    out.xyz = all_xyz

    palette = np.array([[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]], dtype=np.uint8)
    rgb_w = np.full((len(xyz_w), 3), [128, 128, 128], dtype=np.uint8)
    valid_mask = label != -1
    if np.any(valid_mask): rgb_w[valid_mask] = palette[label[valid_mask] % len(palette)]
    tower_color = np.array([173, 216, 230], dtype=np.uint8)
    rgb_t = np.tile(tower_color, (len(xyz_t), 1))
    rgb_all = np.vstack([rgb_t, rgb_w])
    out.red = (rgb_all[:, 0].astype(np.uint16) << 8)
    out.green = (rgb_all[:, 1].astype(np.uint16) << 8)
    out.blue = (rgb_all[:, 2].astype(np.uint16) << 8)

    out.classification = np.hstack([np.full(len(xyz_t), 3, np.uint8), np.full(len(xyz_w), 2, np.uint8)])
    line_id_w = np.zeros(len(xyz_w), dtype=np.uint16)
    if np.any(valid_mask): line_id_w[valid_mask] = label[valid_mask].astype(np.uint16) + 1
    out.line_id = np.hstack([np.zeros(len(xyz_t), np.uint16), line_id_w])
    tower_id_t = np.zeros(len(xyz_t), dtype=np.uint16)
    if tower_segmentation_labels is not None:
        valid_mask_t = tower_segmentation_labels != -1
        tower_id_t[valid_mask_t] = tower_segmentation_labels[valid_mask_t].astype(np.uint16) + 1
    out.tower_id = np.hstack([tower_id_t, np.zeros(len(xyz_w), np.uint16)])

    out.write(cfg["outfile"])
    final_wire_clusters = len(np.unique(label[label != -1]))
    print(f"✓ Saved {cfg['outfile']} — Final wire clusters: {final_wire_clusters}")

if __name__ == "__main__":
    print("Starting Python script execution...")
    main()
    print("Python script execution finished.")

