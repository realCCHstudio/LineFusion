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
        # 移除了不兼容旧版laspy的'encoding'参数
        las = laspy.read(file_path)
        xyz = np.vstack([las.x, las.y, las.z]).T
        process = np.array(las.process) if hasattr(las, 'process') else np.zeros(len(xyz), dtype=int)

        has_rgb = all(d in las.point_format.dimension_names for d in ('red', 'green', 'blue'))
        if has_rgb:
            rgb = np.vstack([las.red >> 8, las.green >> 8, las.blue >> 8]).T.astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)

        return xyz, process, las.classification, rgb
    except Exception as e:
        print(f"加载 LAS 文件时出错: {e}")
        return None, None, None, None

# =========================
# 算法核心函数 (移植自 fit(1).py)
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
        laspy.create().write(output_path)
        sys.exit(1)

    ground_mask = process == 0; tower_mask = process == 1; wire_mask = process >= 2
    xyz_ground, xyz_tower, xyz_wire = xyz[ground_mask], xyz[tower_mask], xyz[wire_mask]
    process_wire = process[wire_mask]

    print(f"地面点: {len(xyz_ground)}, 电塔点: {len(xyz_tower)}, 导线点: {len(xyz_wire)}")
    if len(xyz_wire) == 0:
        print("警告: 没有找到导线点，正在保存空结果文件。")
        laspy.create(point_format=3, file_version="1.2").write(output_path)
        with open(os.path.join(os.path.dirname(output_path), 'span.json'), 'w', encoding='utf-8') as f: json.dump([], f)
        return

    ground_tree = cKDTree(xyz_ground[:, :2]) if len(xyz_ground) > 0 else None
    unique_processes = np.unique(process_wire)
    global_wire_labels = np.full(len(xyz_wire), -1, dtype=int)
    global_label_counter, line_data_list = 0, []

    for process_val in unique_processes:
        print(f"\n处理跨段 process={process_val}...")
        span_mask = process_wire == process_val
        span_xyz, span_indices = xyz_wire[span_mask], np.where(span_mask)[0]
        if len(span_xyz) < cfg["min_cluster_points"]: continue

        print("  去噪...")
        span_xyz_clean, valid_indices = remove_outliers_wire_aware(span_xyz, **cfg)
        if len(span_xyz_clean) < cfg["min_cluster_points"]: continue
        print(f"  去噪后点数: {len(span_xyz_clean)}")

        print("  侧向聚类...")
        main_dir = get_local_direction_fast(span_xyz_clean)
        if main_dir is None: continue
        clusters, basis = split_parallel_wires_by_lateral(span_xyz_clean, main_dir, cfg)
        if not clusters: clusters = [{"indices": np.arange(len(span_xyz_clean))}]
        print(f"  聚类得到 {len(clusters)} 个候选线束")

        valid_span_indices = span_indices[valid_indices]
        for clu in clusters:
            local_idx = clu["indices"]
            sub_points = span_xyz_clean[local_idx]
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
                    _, idx = ground_tree.query(lowest_point[:2], k=min(5, len(xyz_ground)))
                    if isinstance(idx, (np.ndarray, list)) and len(idx) > 0 and np.all(idx < len(xyz_ground)):
                        clearance = lowest_point[2] - np.mean(xyz_ground[idx, 2])

            global_wire_labels[valid_span_indices[local_idx]] = global_label_counter
            line_data_list.append({
                "编号": int(global_label_counter + 1), "所属段": int(process_val),
                "点云数量": int(len(sub_points)), "长度(m)": round(length, 2),
                "弧垂(m)": round(sag, 2), "对地距离(m)": round(clearance, 2),
                "线缆材质": "钢芯铝绞线"
            })
            print(f"    导线 {global_label_counter + 1}: {len(sub_points)} 点, 长度 {length:.1f}m, 弧垂 {sag:.2f}m, 对地距离 {clearance:.2f}m")
            global_label_counter += 1

    json_outfile = os.path.join(os.path.dirname(output_path), 'span.json')
    with open(json_outfile, 'w', encoding='utf-8') as f:
        json.dump(line_data_list, f, indent=4, ensure_ascii=False)
    print(f"\n* 线路属性已保存到 {json_outfile}")

    print("\n正在生成输出文件...")
    hdr = LasHeader(point_format=3, version="1.2")
    hdr.add_extra_dim(ExtraBytesParams(name="line_id", type="uint16"))
    hdr.add_extra_dim(ExtraBytesParams(name="process_id", type="uint16"))
    out = LasData(hdr)
    out.xyz = xyz
    palette = np.array([[31,119,180],[255,127,14],[44,160,44],[214,39,40],[148,103,189]], dtype=np.uint8)
    rgb_all = np.full((len(xyz), 3), 128, dtype=np.uint8)
    rgb_all[ground_mask] = np.array([139, 69, 19])
    rgb_all[tower_mask] = np.array([173, 216, 230])
    valid_wire_mask = global_wire_labels != -1
    if np.any(valid_wire_mask):
        rgb_all[wire_mask][valid_wire_mask] = palette[global_wire_labels[valid_wire_mask] % len(palette)]

    out.red = (rgb_all[:, 0].astype(np.uint16) << 8)
    out.green = (rgb_all[:, 1].astype(np.uint16) << 8)
    out.blue = (rgb_all[:, 2].astype(np.uint16) << 8)

    classification_all = np.zeros(len(xyz), dtype=np.uint8)
    classification_all[ground_mask] = 0
    classification_all[tower_mask] = 2
    classification_all[wire_mask] = 1
    out.classification = classification_all

    out.line_id = np.zeros(len(xyz), dtype=np.uint16)
    if np.any(valid_wire_mask):
        line_ids_full = np.zeros(len(xyz_wire), dtype=np.uint16)
        line_ids_full[valid_wire_mask] = (global_wire_labels[valid_wire_mask] + 1)
        out.line_id[wire_mask] = line_ids_full
    out.process_id = process.astype(np.uint16)

    out.write(output_path)
    final_wire_count = len(np.unique(global_wire_labels[global_wire_labels != -1]))
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
