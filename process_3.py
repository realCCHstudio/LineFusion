# -*- coding: utf-8 -*-
import numpy as np
import laspy
from laspy import LasHeader, LasData, ExtraBytesParams
from scipy.spatial import cKDTree
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import sys
import os

# =========================
# 配置参数 (移植自 fit(1).py 的优化参数)
# =========================
CONFIG = {
    # —— 去噪（电力线友好）——
    "denoise_k_core": 12,
    "denoise_k_lin": 12,
    "denoise_kdist_percentile": 75,
    "denoise_eps_min": 0.15,
    "denoise_eps_max": 3.0,
    "denoise_linearity_tau": 0.80,
    "denoise_keep_ratio_floor": 0.25,

    # —— 侧向聚类（按 v 分簇）——
    "min_cluster_points": 30,
    "cluster_eps_scale": 1.0,
    "cluster_min_samples_factor": 0.5,

    # —— 生长约束 ——
    "angle_deg_max": 12.0,
    "curvature_R_min": 6.0,
    "perp_tolerance_radius": 3.5,
    "base_search_dist": 20.0,
    "max_search_dist": 50.0,
    "step_knn_k": 12,
    "step_scale": 1.0,
    "max_candidates": 300,
    "path_lookback": 15,

    # 缺口桥接
    "gap_bridge_enable": True,
    "max_gap_attempts": 2,
    "gap_expand_factor": 1.8,
    "gap_relax_perp_multiplier": 1.8,
}

# =========================
# I/O
# =========================
def load_las_file(file_path):
    """加载LAS文件"""
    try:
        las = laspy.read(file_path)
        xyz = np.vstack([las.x, las.y, las.z]).T
        # 确保 process 属性存在，如果不存在则初始化为0
        process = np.array(las.process) if hasattr(las, 'process') else np.zeros(len(xyz), dtype=int)

        has_rgb = all(d in las.point_format.dimension_names for d in ('red', 'green', 'blue'))
        if has_rgb:
            rgb = np.vstack([las.red >> 8, las.green >> 8, las.blue >> 8]).T.astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

        # classification 属性也做同样处理
        classification = np.array(las.classification) if hasattr(las, 'classification') else np.zeros(len(xyz), dtype=np.uint8)
        return xyz, process, classification, rgb
    except Exception as e:
        print(f"加载 LAS 文件时出错: {e}")
        return None, None, None, None

# =========================
# 算法核心函数 (与原版一致，无需修改)
# =========================
def remove_outliers_wire_aware(xyz, **cfg):
    n = len(xyz)
    k_core=cfg.get("denoise_k_core", 12); k_lin=cfg.get("denoise_k_lin", 12)
    keep_ratio_floor=cfg.get("denoise_keep_ratio_floor", 0.25)
    eps_max=cfg.get("denoise_eps_max", 3.0)

    if n == 0: return xyz, np.array([], dtype=int)
    if n < max(k_core, k_lin) + 1: return xyz, np.arange(n, dtype=int)

    nbrs = NearestNeighbors(n_neighbors=max(k_core, k_lin), algorithm='kd_tree').fit(xyz)
    dists, inds = nbrs.kneighbors(xyz)

    kth = dists[:, k_core-1]; kth = kth[np.isfinite(kth)]
    if len(kth) == 0: return xyz, np.arange(n, dtype=int)

    eps = float(np.percentile(kth, cfg.get("denoise_kdist_percentile", 75)))
    eps = min(max(eps, cfg.get("denoise_eps_min", 0.15)), eps_max)

    core_mask = dists[:, k_core-1] <= eps
    neighbor_has_core = np.any(core_mask[inds[:, :k_core]], axis=1)
    keep_mask = core_mask | (np.any(dists[:, :k_core] <= eps, axis=1) & neighbor_has_core)

    need_linear_check = np.where(~keep_mask)[0]
    if len(need_linear_check) > 0:
        pts_blocks = xyz[inds[need_linear_check, :k_lin]]
        pts_centered = pts_blocks - pts_blocks.mean(axis=1, keepdims=True)
        covs = np.einsum('mki,mkj->mij', pts_centered, pts_centered) / max(k_lin-1, 1)
        evals = np.linalg.eigvalsh(covs)
        linearity = evals[:, 2] / (np.sum(evals, axis=1) + 1e-12)
        keep_mask[need_linear_check[linearity >= cfg.get("denoise_linearity_tau", 0.80)]] = True

    kept_indices = np.where(keep_mask)[0]
    min_keep = int(np.ceil(keep_ratio_floor * n))
    if len(kept_indices) < min_keep:
        eps2 = min(eps_max, eps * 1.5)
        core_mask2 = dists[:, k_core-1] <= eps2
        neighbor_has_core2 = np.any(core_mask2[inds[:, :k_core]], axis=1)
        keep_mask2 = core_mask2 | (np.any(dists[:, :k_core] <= eps2, axis=1) & neighbor_has_core2)
        kept_indices = np.where(keep_mask2 | keep_mask)[0]

    return xyz[kept_indices], np.unique(kept_indices)

def get_local_direction_fast(points):
    if len(points) < 3: return None
    try:
        _, _, vh = np.linalg.svd(points - np.mean(points, axis=0), full_matrices=False)
        d = vh[0]
        return d / (np.linalg.norm(d) + 1e-9)
    except: return None

def _build_orthonormal_basis(main_dir):
    main_dir /= (np.linalg.norm(main_dir) + 1e-9)
    ref = np.array([0., 0., 1.])
    lat_dir = np.cross(main_dir, ref)
    if np.linalg.norm(lat_dir) < 1e-6:
        ref = np.array([0., 1., 0.])
        lat_dir = np.cross(main_dir, ref)
    lat_dir /= (np.linalg.norm(lat_dir) + 1e-9)
    vert_dir = np.cross(main_dir, lat_dir)
    return main_dir, lat_dir, vert_dir

def _estimate_eps_from_v(v, eps_scale):
    if len(v) < 3: return 0.2
    dv = np.abs(np.diff(np.sort(v))); dv = dv[np.isfinite(dv)]
    if len(dv) == 0: return 0.2
    return max(0.15, float(eps_scale * np.median(dv)))

def split_parallel_wires_by_lateral(span_xyz, main_dir, cfg):
    basis = _build_orthonormal_basis(main_dir)
    v = span_xyz @ basis[1]
    eps = _estimate_eps_from_v(v, cfg["cluster_eps_scale"])
    min_samples = max(int(cfg["min_cluster_points"] * cfg["cluster_min_samples_factor"]), 20)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(v.reshape(-1, 1)).labels_
    clusters = []
    for cid in np.unique(labels[labels != -1]):
        idx = np.where(labels == cid)[0]
        if len(idx) >= cfg["min_cluster_points"]: clusters.append({"indices": idx})
    return clusters, basis

def constrained_polyline_order(sub_points, basis_main_dir, cfg):
    n = len(sub_points)
    if n < 2: return np.arange(n, dtype=int)
    knn_scale = _knn_scale(sub_points, cfg["step_knn_k"]) * cfg["step_scale"]
    tree = cKDTree(sub_points)
    visited = np.zeros(n, dtype=bool)
    proj = sub_points @ basis_main_dir
    start = int(np.argmin(proj))
    order, visited[start], path_dir, gap_attempts_used = [start], True, basis_main_dir.copy(), 0
    while True:
        last, current = order[-1], sub_points[order[-1]]
        if len(order) >= 3:
            recent_pts = sub_points[np.array(order[max(0, len(order) - cfg["path_lookback"]):])]
            est_dir = get_local_direction_fast(recent_pts)
            if est_dir is not None: path_dir = est_dir if np.dot(est_dir, path_dir) >= 0 else -est_dir

        progress = (proj[last] - np.min(proj)) / max(np.ptp(proj), 1e-6)
        r = cfg["base_search_dist"] + progress * (cfg["max_search_dist"] - cfg["base_search_dist"])
        r = max(r, knn_scale * 2.0)
        cand = [c for c in tree.query_ball_point(current, r) if not visited[c]]
        if not cand and cfg["gap_bridge_enable"] and gap_attempts_used < cfg["max_gap_attempts"]:
            gap_attempts_used += 1
            extrap = current + path_dir * (knn_scale * 1.2)
            cand = [c for c in tree.query_ball_point(extrap, r * cfg["gap_expand_factor"]) if not visited[c]]
        if not cand: break

        vecs = sub_points[cand] - current; fwd = vecs @ path_dir
        mask_fwd = fwd > 0
        if not np.any(mask_fwd): break

        cand, vecs = np.array(cand)[mask_fwd], vecs[mask_fwd]
        perp_d = np.linalg.norm(vecs - (vecs @ path_dir)[:, None] * path_dir, axis=1)
        perp_tol = cfg["perp_tolerance_radius"] * (cfg["gap_relax_perp_multiplier"] if gap_attempts_used > 0 else 1.0)
        mask_perp = perp_d <= perp_tol
        if not np.any(mask_perp): break

        cand, vecs = cand[mask_perp], vecs[mask_perp]
        if not cand.size: break
        best = cand[np.argmin(np.linalg.norm(vecs, axis=1))]
        order.append(best); visited[best] = True; gap_attempts_used = 0
        if np.all(visited): break

    return np.array(order, dtype=int)

def _knn_scale(points, k):
    if len(points) <= k: return 1.0
    dists, _ = NearestNeighbors(n_neighbors=k).fit(points).kneighbors(points)
    return float(np.median(dists[:, -1]))

# =========================
# 主流程
# =========================
def main(input_path, output_path):
    cfg = CONFIG
    print(f"正在加载文件: {input_path}")
    xyz, process, classification, rgb = load_las_file(input_path)
    if xyz is None:
        print("错误: 文件加载失败。")
        sys.exit(1)

    # 1. 根据 process 值分离点云
    ground_mask = process == 0
    tower_mask = process == 1
    wire_mask = process >= 2

    xyz_wire = xyz[wire_mask]
    process_wire = process[wire_mask]

    print(f"地面点: {np.sum(ground_mask)}, 电塔点: {np.sum(tower_mask)}, 导线点: {len(xyz_wire)}")
    if len(xyz_wire) == 0:
        print("警告: 没有找到导线点，正在保存原始文件。")
        # 直接写入原始数据
        las = laspy.read(input_path)
        las.write(output_path)
        with open(os.path.join(os.path.dirname(output_path), 'span.json'), 'w', encoding='utf-8') as f:
            json.dump([], f)
        return

    # 准备地面点kd-tree用于计算对地距离
    xyz_ground = xyz[ground_mask]
    ground_tree = cKDTree(xyz_ground[:, :2]) if len(xyz_ground) > 0 else None

    # 初始化一个与导线点数量相同的标签数组，-1代表未分类
    global_wire_labels = np.full(len(xyz_wire), -1, dtype=int)
    global_label_counter = 0
    line_data_list = []

    # 2. 核心处理流程：遍历每个导线段 (span)
    unique_processes = np.unique(process_wire)
    for process_val in unique_processes:
        print(f"\n处理跨段 process={process_val}...")
        span_mask_local = process_wire == process_val
        span_xyz = xyz_wire[span_mask_local]
        # 获取当前段在原始 wire_mask 中的索引
        span_indices_in_wire = np.where(span_mask_local)[0]

        if len(span_xyz) < cfg["min_cluster_points"]: continue

        print("  去噪...")
        span_xyz_clean, valid_indices_in_span = remove_outliers_wire_aware(span_xyz, **cfg)
        if len(span_xyz_clean) < cfg["min_cluster_points"]: continue
        print(f"  去噪后点数: {len(span_xyz_clean)}")

        print("  侧向聚类...")
        main_dir = get_local_direction_fast(span_xyz_clean)
        if main_dir is None: continue
        clusters, basis = split_parallel_wires_by_lateral(span_xyz_clean, main_dir, cfg)
        if not clusters:
            clusters = [{"indices": np.arange(len(span_xyz_clean))}]
        print(f"  聚类得到 {len(clusters)} 个候选线束")

        # 将去噪后的点在“当前段”中的索引，映射回在“所有导线点”中的索引
        valid_span_indices_in_wire = span_indices_in_wire[valid_indices_in_span]

        for clu in clusters:
            local_idx_in_clean = clu["indices"]
            sub_points = span_xyz_clean[local_idx_in_clean]
            order_local = constrained_polyline_order(sub_points, basis[0], cfg)

            length, sag, clearance = 0.0, 0.0, 0.0
            if len(order_local) >= 2:
                seq = sub_points[order_local]
                length = float(np.sum(np.linalg.norm(np.diff(seq, axis=0), axis=1)))
                start_point, end_point = seq[0], seq[-1]
                lowest_point = seq[np.argmin(seq[:, 2])]
                vec_span_xy = (end_point - start_point)[:2]
                if np.linalg.norm(vec_span_xy) > 1e-6:
                    t = np.dot(lowest_point[:2] - start_point[:2], vec_span_xy) / np.dot(vec_span_xy, vec_span_xy)
                    z_line = start_point[2] + np.clip(t, 0, 1) * (end_point[2] - start_point[2])
                    sag = z_line - lowest_point[2]
                if ground_tree is not None:
                    dist, idx = ground_tree.query(lowest_point[:2], k=min(5, len(xyz_ground)))
                    if np.any(np.isfinite(dist)):
                        clearance = lowest_point[2] - np.mean(xyz_ground[idx, 2])

            # 获取这些点在 `global_wire_labels` 数组中对应的最终索引
            final_indices_in_wire = valid_span_indices_in_wire[local_idx_in_clean]
            global_wire_labels[final_indices_in_wire] = global_label_counter

            line_data_list.append({
                "编号": int(global_label_counter + 1), "所属段": int(process_val),
                "点云数量": int(len(sub_points)), "长度(m)": round(length, 2),
                "弧垂(m)": round(sag, 2), "对地距离(m)": round(clearance, 2),
                "线缆材质": "钢芯铝绞线"
            })
            print(f"    导线 {global_label_counter + 1}: {len(sub_points)} 点, 长度 {length:.1f}m, 弧垂 {sag:.2f}m, 对地距离 {clearance:.2f}m")
            global_label_counter += 1

    # 3. 保存JSON结果
    json_outfile = os.path.join(os.path.dirname(output_path), 'span.json')
    with open(json_outfile, 'w', encoding='utf-8') as f:
        json.dump(line_data_list, f, indent=4, ensure_ascii=False)
    print(f"\n* 线路属性已保存到 {json_outfile}")

    # =================================================================
    # 4. 【核心修改】数据重组与LAS文件生成
    # =================================================================
    print("\n正在生成输出文件 (采用数据重组方式)...")

    # 4.1 定义调色板和固定颜色
    palette = np.array([
        [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
        [31, 119, 180], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207]
    ], dtype=np.uint8)
    ground_color = np.array([139, 69, 19], dtype=np.uint8)   # 棕色/橘色
    tower_color = np.array([173, 216, 230], dtype=np.uint8)  # 浅蓝色
    default_color = np.array([128, 128, 128], dtype=np.uint8) # 灰色

    # 4.2 分离出不同类别的点云数据
    xyz_ground = xyz[ground_mask]
    xyz_tower = xyz[tower_mask]

    # 进一步分离导线点：成功分类的 vs. 未成功分类的(噪声)
    processed_wire_mask = global_wire_labels != -1
    unprocessed_wire_mask = global_wire_labels == -1

    xyz_wire_ok = xyz_wire[processed_wire_mask]
    xyz_wire_noise = xyz_wire[unprocessed_wire_mask]

    # 4.3 为每个类别创建对应的属性数组

    # -- 颜色 --
    rgb_ground = np.tile(ground_color, (len(xyz_ground), 1))
    rgb_tower = np.tile(tower_color, (len(xyz_tower), 1))
    rgb_wire_noise = np.tile(default_color, (len(xyz_wire_noise), 1))
    # 为成功分类的导线根据ID应用调色板
    labels_ok = global_wire_labels[processed_wire_mask]
    rgb_wire_ok = palette[labels_ok % len(palette)]

    # -- 分类 (Classification) --
    # 采用标准分类值: 2=地面, 6=建筑(电塔), 14=导线, 1=未分类
    cls_ground = np.full(len(xyz_ground), 2, dtype=np.uint8)
    cls_tower = np.full(len(xyz_tower), 6, dtype=np.uint8)
    cls_wire_ok = np.full(len(xyz_wire_ok), 14, dtype=np.uint8)
    cls_wire_noise = np.full(len(xyz_wire_noise), 1, dtype=np.uint8)

    # -- 线路ID (line_id) --
    lid_ground = np.zeros(len(xyz_ground), dtype=np.uint16)
    lid_tower = np.zeros(len(xyz_tower), dtype=np.uint16)
    lid_wire_ok = (labels_ok + 1).astype(np.uint16) # ID从1开始
    lid_wire_noise = np.zeros(len(xyz_wire_noise), dtype=np.uint16)

    # -- 段ID (process_id) --
    pid_ground = process[ground_mask].astype(np.uint16)
    pid_tower = process[tower_mask].astype(np.uint16)
    pid_wire_ok = process_wire[processed_wire_mask].astype(np.uint16)
    pid_wire_noise = process_wire[unprocessed_wire_mask].astype(np.uint16)

    # 4.4 按相同顺序安全地堆叠所有数据
    # 顺序: 地面 -> 电塔 -> 已分类导线 -> 噪声导线
    final_xyz = np.vstack((xyz_ground, xyz_tower, xyz_wire_ok, xyz_wire_noise))
    final_rgb = np.vstack((rgb_ground, rgb_tower, rgb_wire_ok, rgb_wire_noise))
    final_classification = np.hstack((cls_ground, cls_tower, cls_wire_ok, cls_wire_noise))
    final_line_id = np.hstack((lid_ground, lid_tower, lid_wire_ok, lid_wire_noise))
    final_process_id = np.hstack((pid_ground, pid_tower, pid_wire_ok, pid_wire_noise))

    # 4.5 创建LAS对象并写入数据
    hdr = LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(ExtraBytesParams(name="line_id", type="uint16"))
    hdr.add_extra_dim(ExtraBytesParams(name="process_id", type="uint16"))
    out = LasData(hdr)

    out.xyz = final_xyz
    out.red = final_rgb[:, 0].astype(np.uint16) << 8
    out.green = final_rgb[:, 1].astype(np.uint16) << 8
    out.blue = final_rgb[:, 2].astype(np.uint16) << 8
    out.classification = final_classification
    out.line_id = final_line_id
    out.process_id = final_process_id

    out.write(output_path)
    final_wire_count = len(np.unique(labels_ok))
    print(f"* 保存 {output_path} — 最终导线数: {final_wire_count}")

# =========================
# 脚本入口
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python process_3.py <input_file_path> <output_file_path>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"错误: 输入文件不存在 -> {in_path}")
        sys.exit(1)
    main(in_path, out_path)