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
# 配置参数
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
    "cluster_min_samples_factor": 0.5,   # min_samples = max(int(min_cluster_points*factor), 20)

    # —— 生长约束（本次新增）——
    "angle_deg_max": 12.0,               # 相邻航向最大夹角
    "curvature_R_min": 6.0,              # 曲率半径下限（越小越弯）
    "perp_tolerance_radius": 3.5,        # 垂直偏移容忍（初始）
    "base_search_dist": 20.0,            # 基础搜索距离（初始）
    "max_search_dist": 50.0,             # 最大搜索距离（随进度上限）
    "step_knn_k": 12,                    # 自适应步长：用KNN尺度估计
    "step_scale": 1.0,                   # 步长缩放（乘子）
    "max_candidates": 300,               # 候选点截断
    "path_lookback": 15,                 # 局部方向估计回看长度

    # 缺口桥接
    "gap_bridge_enable": True,
    "max_gap_attempts": 2,
    "gap_expand_factor": 1.8,            # 搜索半径放宽倍数
    "gap_relax_perp_multiplier": 1.8,    # 垂直容忍放宽倍数
}

# =========================
# I/O
# =========================
def load_las_file(file_path):
    """加载LAS文件"""
    try:
        las = laspy.read(file_path)
        xyz = np.vstack([las.x, las.y, las.z]).T
        
        # process_2输出格式：
        # classification: 标准LAS分类 (2=地面, 6=建筑/电塔, 14=导线, 1=未分类)
        # line_id: 跨段编号 (从1开始)  
        # process_id: 类型标识 (0=地面, 1=电塔, 2+=各跨段)
        classification = np.array(las.classification) if hasattr(las, 'classification') else np.zeros(len(xyz), dtype=np.uint8)
        line_id = np.array(las.line_id) if hasattr(las, 'line_id') else np.zeros(len(xyz), dtype=np.uint16)
        process_id = np.array(las.process_id) if hasattr(las, 'process_id') else np.zeros(len(xyz), dtype=np.uint16)
        
        # 处理RGB
        has_rgb = all(d in las.point_format.dimension_names for d in ('red', 'green', 'blue'))
        if has_rgb:
            rgb = np.vstack([las.red >> 8, las.green >> 8, las.blue >> 8]).T.astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
            
        return xyz, classification, line_id, process_id, rgb
    except Exception as e:
        print(f"加载 LAS 文件时出错: {e}")
        return None, None, None, None, None

# =========================
# 去噪（电力线友好）
# =========================
def remove_outliers_wire_aware(xyz, *,
                               k_core=12, k_lin=12,
                               kdist_percentile=75,
                               eps_min=0.15, eps_max=3.0,
                               linearity_tau=0.80,
                               keep_ratio_floor=0.25):
    n = len(xyz)
    if n == 0:
        return xyz, np.array([], dtype=int)
    if n < max(k_core, k_lin) + 1:
        return xyz, np.arange(n, dtype=int)

    nbrs = NearestNeighbors(n_neighbors=max(k_core, k_lin), algorithm='kd_tree').fit(xyz)
    dists, inds = nbrs.kneighbors(xyz)

    kth = dists[:, k_core-1]
    kth = kth[np.isfinite(kth)]
    if len(kth) == 0:
        return xyz, np.arange(n, dtype=int)

    eps = float(np.percentile(kth, kdist_percentile))
    eps = min(max(eps, eps_min), eps_max)

    core_mask = dists[:, k_core-1] <= eps
    within_eps_mask = np.any(dists[:, :k_core] <= eps, axis=1)
    neighbor_has_core = np.any(core_mask[inds[:, :k_core]], axis=1)
    keep_mask = core_mask | (within_eps_mask & neighbor_has_core)

    need_linear_check = np.where(~keep_mask)[0]
    if len(need_linear_check) > 0:
        idx_mat = inds[need_linear_check, :k_lin]
        pts_blocks = xyz[idx_mat]
        pts_centered = pts_blocks - pts_blocks.mean(axis=1, keepdims=True)
        covs = np.einsum('mki,mkj->mij', pts_centered, pts_centered) / max(k_lin-1, 1)
        evals = np.linalg.eigvalsh(covs)
        evals = np.clip(evals, 0.0, None)
        lam1 = evals[:, 2]; lam2 = evals[:, 1]; lam3 = evals[:, 0]
        linearity = lam1 / (lam1 + lam2 + lam3 + 1e-12)
        linear_keep = linearity >= linearity_tau
        keep_mask[need_linear_check[linear_keep]] = True

    kept_indices = np.where(keep_mask)[0]

    min_keep = int(np.ceil(keep_ratio_floor * n))
    if len(kept_indices) < min_keep:
        eps2 = min(eps_max, eps * 1.5)
        core_mask2 = dists[:, k_core-1] <= eps2
        neighbor_has_core2 = np.any(core_mask2[inds[:, :k_core]], axis=1)
        keep_mask2 = core_mask2 | (np.any(dists[:, :k_core] <= eps2, axis=1) & neighbor_has_core2)
        kept_indices = np.where(keep_mask2 | keep_mask)[0]

    kept_indices = np.unique(kept_indices)
    return xyz[kept_indices], kept_indices

# =========================
# 基础工具
# =========================
def get_local_direction_fast(points):
    if len(points) < 3:
        return None
    centered = points - np.mean(points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        d = vh[0]
        n = np.linalg.norm(d)
        return d / n if n > 0 else None
    except:
        return None

def _build_orthonormal_basis(main_dir):
    main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-9)
    ref = np.array([0., 0., 1.])
    lat_dir = np.cross(main_dir, ref)
    if np.linalg.norm(lat_dir) < 1e-6:
        ref = np.array([0., 1., 0.])
        lat_dir = np.cross(main_dir, ref)
    lat_dir /= (np.linalg.norm(lat_dir) + 1e-9)
    vert_dir = np.cross(main_dir, lat_dir)
    vert_dir /= (np.linalg.norm(vert_dir) + 1e-9)
    return main_dir, lat_dir, vert_dir

def _estimate_eps_from_v(v, eps_scale):
    if len(v) < 3:
        return 0.2
    vv = np.sort(v)
    dv = np.abs(vv[1:] - vv[:-1])
    dv = dv[np.isfinite(dv)]
    if len(dv) == 0:
        return 0.2
    scale = np.median(dv)
    return max(0.15, float(eps_scale * scale))

def split_parallel_wires_by_lateral(span_xyz, main_dir, cfg):
    main_dir, lat_dir, vert_dir = _build_orthonormal_basis(main_dir)
    u = span_xyz @ main_dir
    v = span_xyz @ lat_dir
    w = span_xyz @ vert_dir

    eps = _estimate_eps_from_v(v, cfg["cluster_eps_scale"])
    min_samples = max(int(cfg["min_cluster_points"] * cfg["cluster_min_samples_factor"]), 20)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(v.reshape(-1, 1)).labels_

    clusters = []
    for cid in np.unique(labels):
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < cfg["min_cluster_points"]:
            continue
        clusters.append({"indices": idx, "u": u[idx], "v": v[idx], "w": w[idx]})
    return clusters, (main_dir, lat_dir, vert_dir)

def _knn_scale(points, k):
    if len(points) <= k:
        return 1.0
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points)
    dists, _ = nbrs.kneighbors(points)
    return float(np.median(dists[:, -1]))

def _radius_of_curvature(p0, p1, p2):
    a = np.linalg.norm(p1 - p0)
    b = np.linalg.norm(p2 - p1)
    c = np.linalg.norm(p2 - p0)
    s = 0.5 * (a + b + c)
    area_sq = max(s * (s - a) * (s - b) * (s - c), 0.0)
    if area_sq <= 1e-12:
        return np.inf
    area = np.sqrt(area_sq)
    return (a * b * c) / (4.0 * area)

# =========================
# 受约束的生长（本次增强）
# =========================
def constrained_polyline_order(sub_points, basis_main_dir, cfg):
    """
    在单个侧向簇里，以受约束生长得到"有序折线"的点序索引（局部索引）。
    - 角度阈值/曲率半径
    - 自适应步长（KNN尺度）
    - 缺口桥接（有限次放宽搜索/垂距）
    若中途失败，返回当前已得到的序列。
    """
    n = len(sub_points)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # 自适应尺度与数据结构
    knn_scale = _knn_scale(sub_points, cfg["step_knn_k"]) * cfg["step_scale"]
    tree = cKDTree(sub_points)
    visited = np.zeros(n, dtype=bool)

    # 从一端开始：按主方向投影最小的点作 seed
    proj = sub_points @ (basis_main_dir / (np.linalg.norm(basis_main_dir) + 1e-9))
    start = int(np.argmin(proj))
    order = [start]
    visited[start] = True

    # 生长参数
    angle_max = np.deg2rad(cfg["angle_deg_max"])
    R_min = cfg["curvature_R_min"]
    perp_tol = cfg["perp_tolerance_radius"]
    base_r = cfg["base_search_dist"]
    max_r = cfg["max_search_dist"]

    # 初始方向：沿主方向
    path_dir = basis_main_dir / (np.linalg.norm(basis_main_dir) + 1e-9)

    # 预置：近邻窗口用于方向估计
    lookback = cfg["path_lookback"]

    # 生长循环
    gap_attempts_used = 0
    while True:
        last = order[-1]
        current = sub_points[last]

        # 近邻方向回看
        if len(order) >= 3:
            i0 = max(0, len(order) - lookback)
            recent = sub_points[np.array(order[i0:])]
            est = get_local_direction_fast(recent)
            if est is not None:
                # 保持方向一致性
                if np.dot(est, path_dir) < 0:
                    est = -est
                path_dir = est

        # 动态搜索半径（随进度）+ 自适应尺度
        progress = (proj[last] - np.min(proj)) / max(np.ptp(proj), 1e-6)
        r = base_r + progress * (max_r - base_r)
        r = max(r, knn_scale * 2.0)  # 至少覆盖两个步长
        r = float(min(r, max_r))

        # 查询候选
        cand = tree.query_ball_point(current, r)
        cand = [c for c in cand if not visited[c] and c != last]
        if len(cand) == 0:
            # 尝试缺口桥接
            if cfg["gap_bridge_enable"] and gap_attempts_used < cfg["max_gap_attempts"]:
                gap_attempts_used += 1
                # 外推一点：沿 path_dir 走一小步，再放宽搜索
                extrap = current + path_dir * (knn_scale * 1.2)
                cand = tree.query_ball_point(extrap, r * cfg["gap_expand_factor"])
                # 放宽垂距条件
                cand_perp_tol = perp_tol * cfg["gap_relax_perp_multiplier"]
                # 转入统一筛选逻辑（下面会应用）
            else:
                break

        # 矢量化筛选
        cand_pts = sub_points[cand]
        vecs = cand_pts - current
        fwd = vecs @ path_dir
        mask_fwd = fwd > 0
        if not np.any(mask_fwd):
            # 无前进候选：尝试一次外推已做过，不再循环
            break

        cand_idx1 = np.where(mask_fwd)[0]
        vecs1 = vecs[cand_idx1]
        fwd1 = fwd[cand_idx1]

        # 垂直距离
        perp = vecs1 - fwd1[:, None] * path_dir
        perp_d = np.linalg.norm(perp, axis=1)
        cand_perp_tol = perp_tol if gap_attempts_used == 0 else perp_tol * cfg["gap_relax_perp_multiplier"]
        mask_perp = perp_d <= cand_perp_tol
        if not np.any(mask_perp):
            # 即便桥接放宽仍失败
            break

        cand_idx2 = cand_idx1[mask_perp]
        cand_pts2 = cand_pts[cand_idx2]
        dists2 = np.linalg.norm(vecs1[mask_perp], axis=1)

        # 航向角约束：与上一段夹角 <= angle_max
        if len(order) >= 2:
            prev = sub_points[order[-2]]
            prev_dir = sub_points[order[-1]] - prev
            if np.linalg.norm(prev_dir) > 1e-9:
                prev_dir = prev_dir / np.linalg.norm(prev_dir)
                dirs_to_cand = cand_pts2 - current
                dirs_to_cand /= (np.linalg.norm(dirs_to_cand, axis=1, keepdims=True) + 1e-9)
                cosang = np.clip(dirs_to_cand @ prev_dir, -1, 1)
                ang = np.arccos(cosang)
                mask_ang = ang <= angle_max
            else:
                mask_ang = np.ones(len(cand_idx2), dtype=bool)
        else:
            mask_ang = np.ones(len(cand_idx2), dtype=bool)

        if not np.any(mask_ang):
            # 角度全不满足时，保留一半最接近的候选以避免卡死
            keep_idx = np.argsort(dists2)[:max(1, len(dists2)//2)]
        else:
            keep_idx = np.where(mask_ang)[0]

        cand_idx3 = cand_idx2[keep_idx]
        cand_pts3 = cand_pts2[keep_idx]
        dists3 = dists2[keep_idx]

        # 曲率约束：用 (p_{-2}, p_{-1}, cand) 估计半径
        if len(order) >= 2 and len(cand_idx3) > 0:
            p0 = sub_points[order[-2]]
            p1 = sub_points[order[-1]]
            R = np.array([_radius_of_curvature(p0, p1, q) for q in cand_pts3])
            mask_R = R >= R_min
            if not np.any(mask_R):
                # 如果都太弯：保留 R 最大的一半
                top_idx = np.argsort(-R)[:max(1, len(R)//2)]
                cand_idx4 = cand_idx3[top_idx]
                dists4 = dists3[top_idx]
            else:
                cand_idx4 = cand_idx3[mask_R]
                dists4 = dists3[mask_R]
        else:
            cand_idx4 = cand_idx3
            dists4 = dists3

        if len(cand_idx4) == 0:
            break

        # 候选截断 + 选最近
        if len(cand_idx4) > CONFIG["max_candidates"]:
            top = np.argpartition(dists4, CONFIG["max_candidates"])[:CONFIG["max_candidates"]]
            cand_idx4 = cand_idx4[top]
            dists4 = dists4[top]

        best_local = cand_idx4[int(np.argmin(dists4))]
        best = cand[best_local]
        order.append(best)
        visited[best] = True
        # 更新方向（在下一轮由 recent 点估出来）
        gap_attempts_used = 0  # 一旦成功连上，重置桥接尝试计数

        if np.all(visited):
            break

    return np.array(order, dtype=int)

# =========================
# 主流程
# =========================
def main(input_path, output_path):
    cfg = CONFIG
    print(f"正在加载文件: {input_path}")
    
    # 加载输入文件
    xyz, classification, line_id, process_id, rgb = load_las_file(input_path)

    if xyz is None:
        print("错误: 文件加载失败。")
        sys.exit(1)
    
    print(f"总点数: {len(xyz)}")

    # 根据process_2的输出格式分离点云
    ground_mask = process_id == 0  # 地面点
    tower_mask = process_id == 1   # 电塔点  
    wire_mask = process_id >= 2    # 导线点（跨段）

    xyz_ground = xyz[ground_mask]
    xyz_tower = xyz[tower_mask]
    xyz_wire = xyz[wire_mask]
    line_id_wire = line_id[wire_mask]  # 导线点的跨段ID
    process_id_wire = process_id[wire_mask]  # 导线点的process_id

    print(f"地面点: {len(xyz_ground)}, 电塔点: {len(xyz_tower)}, 导线点: {len(xyz_wire)}")
    if len(xyz_wire) == 0:
        print("警告: 没有找到导线点，正在保存原始文件。")
        # 直接写入原始数据
        las = laspy.read(input_path)
        las.write(output_path)
        # 生成空的span.json
        span_json_path = os.path.join(os.path.dirname(output_path), 'span.json')
        with open(span_json_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=4, ensure_ascii=False)
        return

    # 准备地面点kd-tree用于计算对地距离
    ground_tree = cKDTree(xyz_ground[:, :2]) if len(xyz_ground) > 0 else None

    # 获取所有唯一的跨段ID
    unique_span_ids = np.unique(line_id_wire[line_id_wire > 0])
    print(f"发现 {len(unique_span_ids)} 个跨段: {unique_span_ids}")

    # 按跨段点数排序（从小到大处理）
    span_sizes = [(span_id, np.sum(line_id_wire == span_id)) for span_id in unique_span_ids]
    span_sizes.sort(key=lambda x: x[1])

    global_wire_labels = np.full(len(xyz_wire), -1, dtype=int)
    global_label_counter = 0
    line_data_list = []

    for span_id, point_count in span_sizes:
        print(f"\n处理跨段 span_id={span_id} (点数: {point_count})...")

        span_mask = line_id_wire == span_id
        span_xyz = xyz_wire[span_mask]
        span_indices = np.where(span_mask)[0]

        if len(span_xyz) < cfg["min_cluster_points"]:
            print(f"  跳过: 点数不足 {cfg['min_cluster_points']}")
            continue

        print("  去噪（电力线友好）...")
        span_xyz_clean, valid_indices = remove_outliers_wire_aware(
            span_xyz,
            k_core=cfg["denoise_k_core"],
            k_lin=cfg["denoise_k_lin"],
            kdist_percentile=cfg["denoise_kdist_percentile"],
            eps_min=cfg["denoise_eps_min"], eps_max=cfg["denoise_eps_max"],
            linearity_tau=cfg["denoise_linearity_tau"],
            keep_ratio_floor=cfg["denoise_keep_ratio_floor"]
        )
        if len(span_xyz_clean) < cfg["min_cluster_points"]:
            print("  跳过: 去噪后点数不足")
            continue
        print(f"  去噪后点数: {len(span_xyz_clean)}")

        print("  电力线分根（侧向聚类）...")
        main_dir = get_local_direction_fast(span_xyz_clean)
        if main_dir is None:
            print("  警告: 主方向估计失败，跳过该跨段")
            continue

        clusters, basis = split_parallel_wires_by_lateral(span_xyz_clean, main_dir, cfg)
        if len(clusters) == 0:
            print("  未检测到多根导线，作为单根处理")
            clusters = [{"indices": np.arange(len(span_xyz_clean)),
                         "u": (span_xyz_clean @ basis[0]),
                         "v": (span_xyz_clean @ basis[1]),
                         "w": (span_xyz_clean @ basis[2])}]

        print(f"  检测到 {len(clusters)} 根平行导线")

        valid_span_indices = span_indices[valid_indices]

        for c_idx, clu in enumerate(clusters, 1):
            local_idx = clu["indices"]
            if len(local_idx) < cfg["min_cluster_points"]:
                continue

            sub_points = span_xyz_clean[local_idx]

            # —— 受约束生长得到"顺序折线"用于鲁棒长度估计 ——
            order_local = constrained_polyline_order(sub_points, basis[0], cfg)
            
            # 计算长度、弧垂、对地距离
            length, sag, clearance = 0.0, 0.0, 0.0
            if len(order_local) >= 2:
                seq = sub_points[order_local]
                length = float(np.sum(np.linalg.norm(np.diff(seq, axis=0), axis=1)))
                
                # 计算弧垂
                start_point, end_point = seq[0], seq[-1]
                lowest_point = seq[np.argmin(seq[:, 2])]
                vec_span_xy = (end_point - start_point)[:2]
                if np.linalg.norm(vec_span_xy) > 1e-6:
                    t = np.dot(lowest_point[:2] - start_point[:2], vec_span_xy) / np.dot(vec_span_xy, vec_span_xy)
                    z_line = start_point[2] + np.clip(t, 0, 1) * (end_point[2] - start_point[2])
                    sag = z_line - lowest_point[2]
                
                # 计算对地距离
                if ground_tree is not None:
                    dist, idx = ground_tree.query(lowest_point[:2], k=min(5, len(xyz_ground)))
                    if np.any(np.isfinite(dist)):
                        clearance = lowest_point[2] - np.mean(xyz_ground[idx, 2])
            else:
                # 退化到按 u 排序的长度估计
                proj_u = sub_points @ (basis[0] / (np.linalg.norm(basis[0]) + 1e-9))
                seq = sub_points[np.argsort(proj_u)]
                length = float(np.sum(np.linalg.norm(np.diff(seq, axis=0), axis=1)))

            # 回填全局标签（每根导线一个独立的 line_id）
            global_indices = valid_span_indices[local_idx]
            global_wire_labels[global_indices] = global_label_counter

            # 生成JSON格式数据
            line_data_list.append({
                "编号": int(global_label_counter + 1), 
                "所属段": int(span_id),  # 原始跨段ID
                "点云数量": int(len(sub_points)), 
                "长度(m)": round(length, 2),
                "弧垂(m)": round(sag, 2), 
                "对地距离(m)": round(clearance, 2),
                "线缆材质": "钢芯铝绞线"
            })
            print(f"    导线 {global_label_counter + 1}: {len(sub_points)} 点, 长度 {length:.1f}m, 弧垂 {sag:.2f}m, 对地距离 {clearance:.2f}m")
            global_label_counter += 1

    # —— JSON 输出——
    span_json_path = os.path.join(os.path.dirname(output_path), 'span.json')
    with open(span_json_path, 'w', encoding='utf-8') as f:
        json.dump(line_data_list, f, indent=4, ensure_ascii=False)
    print(f"\n* 线路属性已保存到 {span_json_path}")

    # =================================================================
    # 数据重组与LAS文件生成
    # =================================================================
    print("\n正在生成输出文件...")

    # 定义调色板和固定颜色
    palette = np.array([
        [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
        [31, 119, 180], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207]
    ], dtype=np.uint8)
    ground_color = np.array([150, 150, 150], dtype=np.uint8)   # 棕色
    tower_color = np.array([173, 216, 230], dtype=np.uint8)  # 浅蓝色
    default_color = np.array([128, 128, 128], dtype=np.uint8) # 灰色

    # 分离出不同类别的点云数据
    processed_wire_mask = global_wire_labels != -1
    unprocessed_wire_mask = global_wire_labels == -1

    xyz_wire_ok = xyz_wire[processed_wire_mask]
    xyz_wire_noise = xyz_wire[unprocessed_wire_mask]

    # 为每个类别创建对应的属性数组
    # -- 颜色 --
    rgb_ground = np.tile(ground_color, (len(xyz_ground), 1))
    rgb_tower = np.tile(tower_color, (len(xyz_tower), 1))
    rgb_wire_noise = np.tile(default_color, (len(xyz_wire_noise), 1))
    # 为成功分类的导线根据ID应用调色板
    labels_ok = global_wire_labels[processed_wire_mask]
    rgb_wire_ok = palette[labels_ok % len(palette)]

    # -- 分类 (Classification) --
    # 保持原有分类
    cls_ground = np.full(len(xyz_ground), 2, dtype=np.uint8)
    cls_tower = np.full(len(xyz_tower), 6, dtype=np.uint8)
    cls_wire_ok = np.full(len(xyz_wire_ok), 14, dtype=np.uint8)
    cls_wire_noise = np.full(len(xyz_wire_noise), 1, dtype=np.uint8)

    # -- 线路ID (line_id) --
    lid_ground = np.zeros(len(xyz_ground), dtype=np.uint16)
    lid_tower = np.zeros(len(xyz_tower), dtype=np.uint16)
    lid_wire_ok = (labels_ok + 1).astype(np.uint16) # 分根后的新ID从1开始
    lid_wire_noise = np.zeros(len(xyz_wire_noise), dtype=np.uint16)

    # -- 段ID (process_id) -- 保留原始的跨段信息
    pid_ground = process_id[ground_mask].astype(np.uint16)
    pid_tower = process_id[tower_mask].astype(np.uint16)
    pid_wire_ok = process_id_wire[processed_wire_mask].astype(np.uint16)
    pid_wire_noise = process_id_wire[unprocessed_wire_mask].astype(np.uint16)

    # 按相同顺序安全地堆叠所有数据
    # 顺序: 地面 -> 电塔 -> 已分类导线 -> 噪声导线
    final_xyz = np.vstack((xyz_ground, xyz_tower, xyz_wire_ok, xyz_wire_noise))
    final_rgb = np.vstack((rgb_ground, rgb_tower, rgb_wire_ok, rgb_wire_noise))
    final_classification = np.hstack((cls_ground, cls_tower, cls_wire_ok, cls_wire_noise))
    final_line_id = np.hstack((lid_ground, lid_tower, lid_wire_ok, lid_wire_noise))
    final_process_id = np.hstack((pid_ground, pid_tower, pid_wire_ok, pid_wire_noise))

    # 创建LAS对象并写入数据
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
    final_wire_count = len(np.unique(labels_ok)) if len(labels_ok) > 0 else 0
    print(f"* 保存 {output_path} — 最终导线数: {final_wire_count}")

# =========================
# 脚本入口
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python fit.py <input_file_path> <output_file_path>")
        sys.exit(1)
    
    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"错误: 输入文件不存在 -> {in_path}")
        sys.exit(1)
    
    main(in_path, out_path)
