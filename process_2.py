#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
span_segmentation_visualizer.py — (V7.1, 快速防串色：邻接+矢量化)
================================================================
在 V7.0 的基础上优化：
- 防串色仅比较“邻居跨段”（共享电塔或端点接近）
- 批量矢量化计算分数，避免逐点/逐跨内层循环
- 其余流程与 V7.0 一致（两阶段吸附、s 轴 close、自适应带宽等）
依赖：numpy, scipy, open3d, laspy
"""

import colorsys
from collections import deque, defaultdict
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import laspy
from laspy import LasHeader, LasData, ExtraBytesParams
import sys
import os
import json

CONFIG = {


    # =============== 电塔检测 ===============
    "tower_radius": 1.0,
    "tower_density": 40,
    "tower_hmin": 10.0,
    "tower_expand": 3.0,

    # =============== 跨段候选 ===============
    "tower_cluster_eps": 40.0,
    "topology_k_neighbors": 4,
    "span_validation_min_points": 45,
    "tower_cluster_min_points": 20,
    "protection_radius": 20.0,
    "span_width_buffer": 20.0,

    # =============== 去噪 ===============
    "noise_nb": 10,
    "noise_r": 1.5,

    # =============== 线性度 & 缺口 & 分层 ===============
    "linearity_L_min": 0.55,
    "linearity_k": 12,
    "gap_max_mult": 8.0,
    "gap_search_r_mult": 3.0,
    "max_layers": 6,

    # =============== 回收/最小点数自适应 ===============
    "recover_width_scale": 0.03,
    "recover_base_enlarge": 1.25,
    "pca_refine_extra_margin": 2.0,
    "min_points_per_meter_factor": 0.25,

    # =============== 地面感知 ===============
    "ground_relax_dz": 3.0,
    "relax_width_gain": 1.5,
    "relax_Lmin_drop": 0.1,

    # =============== RANSAC 抛物线 ===============
    "ransac_iters": 200,
    "ransac_band_sigma": 2.5,
    "ransac_min_frac": 0.15,

    # =============== 两阶段吸附（阶段1/2） ===============
    "snap1_center_r_mult": 1.8,
    "snap1_band_sigma": 2.8,
    "snap2_center_r_mult": 2.8,
    "snap2_band_sigma": 3.8,
    "snap_abs_floor": 0.6,
    "snap_iters": 2,
    "snap_residual_weight": 0.7,

    # =============== s 轴形态学 close ===============
    "close_gap_mult": 6.0,
    "close_search_mult": 2.5,

    # =============== 防串色（快速版） ===============
    "anti_bleed_margin": 0.2,        # 分数需降低超过 20% 才换色
    "neighbor_end_dist": 50.0,        # 端点接近阈值（米）：控制邻接范围
}

# ----------------------- 小工具 -----------------------
def generate_distinct_colors(n):
    if n == 0: return np.array([])
    hues = np.linspace(0, 1, n, endpoint=False)
    cols = (np.array([colorsys.hls_to_rgb(h, 0.6, 0.8) for h in hues]) * 255).astype(np.uint8)
    return cols

def estimate_point_spacing(xyz, k=8):
    if len(xyz) < k+1: return 0.5
    d, _ = cKDTree(xyz).query(xyz, k=k+1)
    return float(np.median(d[:,1:]))

# ----------------------- IO & 预处理 -----------------------
def load_cloud(path):
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T
    cls = las.classification
    return xyz, cls

def remove_outliers(xyz, nb, r):
    if nb <= 0 or r <= 0 or len(xyz) < nb: return xyz
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    _, idx = pcd.remove_radius_outlier(nb_points=nb, radius=r)
    return xyz[idx] if len(idx) else xyz

def tower_mask(xyz, radius, density, hmin, expand):
    tree = cKDTree(xyz[:, :2])
    cnt = tree.query_ball_point(xyz[:, :2], radius, return_length=True)
    dense = cnt >= density
    neigh = tree.query_ball_point(xyz[:, :2], radius)
    zmax = np.array([xyz[n][:,2].max() if n else xyz[i,2] for i,n in enumerate(neigh)])
    tall = (zmax - xyz[:,2]) >= hmin
    m = dense & tall
    if m.any():
        base = xyz[m][:,2].min()
        xy_t = xyz[m][:, :2]
        dist = cKDTree(xy_t).query(xyz[:, :2], k=1, distance_upper_bound=expand)[0]
        m |= (dist < expand) & (xyz[:,2] >= base)
    return m

# ----------------------- 电塔聚类 -----------------------
def get_tower_centers(xyz_t, cfg):
    min_pts = cfg.get("tower_cluster_min_points", 20)
    if len(xyz_t) < min_pts: return np.empty((0,3)), None
    tree = cKDTree(xyz_t[:, :2])
    visited = np.zeros(len(xyz_t), bool)
    clusters = []
    for i in range(len(xyz_t)):
        if visited[i]: continue
        q = deque([i]); visited[i] = True; comp=[]
        while q:
            cur=q.popleft(); comp.append(cur)
            for nb in tree.query_ball_point(xyz_t[cur,:2], cfg['tower_cluster_eps']):
                if not visited[nb]: visited[nb]=True; q.append(nb)
        if len(comp) >= min_pts: clusters.append(comp)
    centers=[]; labels=np.full(len(xyz_t), -1, int)
    for cid, idxs in enumerate(clusters):
        pts = xyz_t[idxs]
        xy = (pts[:,:2].min(0)+pts[:,:2].max(0))/2
        z  = pts[:,2].mean()
        centers.append([xy[0],xy[1],z]); labels[idxs]=cid
    return np.asarray(centers,float), labels

# ----------------------- 地面感知 -----------------------
def build_ground_model(xyz_ground):
    if len(xyz_ground)==0:
        return lambda xy: np.full(len(xy), -1e9), lambda pts: np.full(len(pts), 1e9)
    tree = cKDTree(xyz_ground[:,:2])
    gz  = xyz_ground[:,2]
    def ground_z(xy, k=6):
        d, idx = tree.query(xy, k=min(k, len(xyz_ground)))
        if np.ndim(idx)==0: return gz[idx]
        return np.mean(gz[idx], axis=-1)
    def dz_to_ground(pts, k=6):
        return pts[:,2] - ground_z(pts[:,:2], k=k)
    return ground_z, dz_to_ground

# ----------------------- 几何基元 -----------------------
def span_local_axes(a,b):
    u=(b-a)/(np.linalg.norm(b-a)+1e-9)
    z=np.array([0,0,1.0],float)
    n=np.cross(u,z)
    if np.linalg.norm(n)<1e-6:
        n=np.cross(u, np.array([0,1.0,0],float))
    n/=np.linalg.norm(n)+1e-9
    return u,n,z

def centerline_window(a,b, pr):
    L=np.linalg.norm(b-a)
    pr=min(max(pr,0.0), L*0.45)
    u=(b-a)/L; a2=a+u*pr; b2=b-u*pr
    L2=np.linalg.norm(b2-a2); u2=(b2-a2)/(L2+1e-9)
    return a2, b2, u2, L2

def centerline_dist(points, a,b, pr):
    a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
    ap=points-a2; t=ap@u2; tclip=np.clip(t,0.0,L2)
    closest=a2+np.outer(tclip,u2)
    return np.linalg.norm(points-closest, axis=1), t, L2, u2, a2

def project_to_span(points, a,b, pr):
    d,t,L2,u2,a2 = (*centerline_dist(points,a,b,pr),)
    _,n,_=span_local_axes(a,b)
    h=(points-a2)@n
    return t, h, L2, u2, n, a2

# ----------------------- 线性度 -----------------------
def compute_local_linearity(points, k=12):
    if len(points)<k+1:
        return np.ones(len(points)), np.tile(np.array([1,0,0]), (len(points),1))
    tree=cKDTree(points)
    L=np.zeros(len(points)); U=np.zeros((len(points),3))
    for i in range(len(points)):
        _,idx=tree.query(points[i], k=k)
        nb=points[idx]-points[i]
        cov=nb.T@nb/max(len(idx)-1,1)
        w,v=np.linalg.eigh(cov)
        lam1,lam2=w[2],w[1]
        L[i]=0.0 if lam1<1e-12 else (lam1-lam2)/lam1
        U[i]=v[:,2]
    return L,U

def filter_by_linearity(points, idx, L_min=0.6, k=12):
    if len(idx)==0: return idx
    sub=points[idx]; L,_=compute_local_linearity(sub,k=k)
    return idx[np.where(L>=L_min)[0]]

# ----------------------- 候选跨段发现 -----------------------
def adaptive_min_points_for_span(L, spacing, base_min, factor):
    eps=1e-6
    return int(max(base_min, np.ceil(L/max(spacing,eps)*factor)))

def points_in_tube_indices(points, a, b, base_width, width_scale, pr):
    a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
    radius=base_width+width_scale*L2
    ap=points-a2; t=ap@u2; tclip=np.clip(t,0.0,L2)
    closest=a2+np.outer(tclip,u2)
    dist=np.linalg.norm(points-closest, axis=1)
    return np.where((t>=0.0)&(t<=L2)&(dist<=radius))[0]

def discover_and_assign_spans_adaptive(tower_centers, xyz_w, cfg, dz_to_ground):
    if len(tower_centers)<2 or len(xyz_w)==0:
        return [], {}, {"spacing":1.0,"base_width":1.0,"width_scale":0.02}

    spacing = estimate_point_spacing(xyz_w)
    base_width = max(cfg.get('span_width_buffer',20.0)*0.25, spacing*2.0)
    width_scale = 0.02
    k = min(cfg['topology_k_neighbors']+1, len(tower_centers))
    tower_tree = cKDTree(tower_centers)

    base_min = cfg['span_validation_min_points']
    factor   = cfg.get("min_points_per_meter_factor",0.25)

    Lmin0 = cfg.get("linearity_L_min",0.55)
    kPCA  = cfg.get("linearity_k",12)
    relax_dz  = cfg.get("ground_relax_dz",3.0)
    relax_gain= cfg.get("relax_width_gain",1.5)
    relax_drop= cfg.get("relax_Lmin_drop",0.1)

    candidate_spans=[]; per_span_indices={}
    for i, ps in enumerate(tower_centers):
        _, nbr = tower_tree.query(ps, k=k)
        for j in nbr:
            if i==j: continue
            a, b = tower_centers[i], tower_centers[j]
            L = np.linalg.norm(b-a)
            if L < cfg['protection_radius']*2: continue

            min_pts = adaptive_min_points_for_span(L, spacing, base_min, factor)

            bw = base_width; ws = width_scale; Lmin = Lmin0
            mid = (a+b)/2.0
            dz_mid = dz_to_ground(mid.reshape(1,3))[0]
            if dz_mid < relax_dz:
                bw *= relax_gain
                Lmin = max(0.0, Lmin - relax_drop)

            idx = points_in_tube_indices(xyz_w, a,b, bw, ws, cfg['protection_radius'])
            idx = filter_by_linearity(xyz_w, idx, L_min=Lmin, k=kPCA)

            if len(idx) >= min_pts:
                key=tuple(sorted((i,j)))
                candidate_spans.append(key)
                per_span_indices[key]=idx

    candidate_spans=sorted(set(candidate_spans))
    if not candidate_spans:
        return [], {}, {"spacing":spacing,"base_width":base_width,"width_scale":width_scale}

    # 冲突消解：中心线距离最小
    point_to={}
    dcache={}
    for key in candidate_spans:
        i,j = key
        idx = per_span_indices[key]
        d,_,_,_,_ = centerline_dist(xyz_w[idx], tower_centers[i], tower_centers[j], cfg['protection_radius'])
        dcache[key]=(idx,d)
        for p in idx:
            point_to.setdefault(int(p), []).append(key)

    assign={}
    for p, cand_keys in point_to.items():
        best=None; bd=np.inf
        for key in cand_keys:
            idx,d=dcache[key]; where=np.where(idx==p)[0]
            if len(where)==0: continue
            val=d[where[0]]
            if val<bd: bd=val; best=key
        if best is not None: assign[p]=best

    out={}
    for key in candidate_spans:
        mem=[p for p,s in assign.items() if s==key]
        a,b=tower_centers[key[0]], tower_centers[key[1]]
        L=np.linalg.norm(b-a)
        min_pts=adaptive_min_points_for_span(L, spacing, base_min, factor)
        if len(mem)>=min_pts:
            out[key]=np.asarray(mem,int)

    return list(out.keys()), out, {"spacing":spacing,"base_width":base_width,"width_scale":width_scale}

# ----------------------- 1D KMeans（无 sklearn） -----------------------
def kmeans_1d(values, k, max_iter=50, seed=0):
    v = np.asarray(values).reshape(-1)
    n = len(v)
    if k <= 1 or n == 0:
        return np.zeros(n, dtype=int), np.array([np.median(v) if n else 0.0])
    qs = np.linspace(0, 100, k, endpoint=False) + 50.0/k
    centers = np.percentile(v, qs)
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = np.abs(v[:, None] - centers[None, :])
        new_labels = np.argmin(d, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
        for j in range(k):
            sel = (labels == j)
            centers[j] = np.mean(v[sel]) if np.any(sel) else float(v[rng.integers(0, n)])
    order = np.argsort(centers)
    remap = {old:new for new, old in enumerate(order)}
    centers = centers[order]
    labels = np.array([remap[x] for x in labels], dtype=int)
    return labels, centers

def split_parallel_layers(points, a,b, max_layers=6):
    if len(points)<10: return [np.arange(len(points))]
    u=(b-a)/(np.linalg.norm(b-a)+1e-9)
    z=np.array([0,0,1.0],float); n=np.cross(u,z)
    if np.linalg.norm(n)<1e-6: n=np.cross(u, np.array([0,1.0,0],float))
    n/=np.linalg.norm(n)+1e-9
    h=(points-a)@n
    hist,_=np.histogram(h, bins=20)
    peaks=(hist[1:-1]>hist[:-2]) & (hist[1:-1]>hist[2:])
    est=int(np.sum(peaks))+1
    n_layers=int(np.clip(est,1,max_layers))
    if n_layers==1: return [np.arange(len(points))]
    labels, _ = kmeans_1d(h, n_layers, max_iter=50, seed=0)
    layers=[np.where(labels==k)[0] for k in range(n_layers)]
    layers=[idx for idx in layers if len(idx)>0]
    layers.sort(key=lambda idx: np.median(h[idx]))
    return layers

# ----------------------- 缺口桥接 -----------------------
def bridge_gaps_along_centerline(points, a,b, pr, assigned_idx, spacing, max_gap_mult=8.0, search_radius_mult=3.0):
    if len(assigned_idx)<5: return np.array([],int)
    a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
    P=points[assigned_idx]; t=(P-a2)@u2
    t_sorted=np.sort(t)
    gaps=[]
    for i in range(len(t_sorted)-1):
        if (t_sorted[i+1]-t_sorted[i])>max_gap_mult*spacing:
            gaps.append((t_sorted[i], t_sorted[i+1]))
    if not gaps: return np.array([],int)
    aset=set(assigned_idx.tolist())
    mask=np.ones(len(points),bool)
    if aset: mask[list(aset)]=False
    C=points[mask]; cand_idx=np.where(mask)[0]
    ap=C-a2; tC=ap@u2; tclip=np.clip(tC,0.0,L2)
    closest=a2+np.outer(tclip,u2)
    d=np.linalg.norm(C-closest, axis=1)
    pick=[]; sr=search_radius_mult*spacing
    for t0,t1 in gaps:
        m=(tC>=t0)&(tC<=t1)&(d<=sr)
        if np.any(m): pick.extend(cand_idx[np.where(m)[0]].tolist())
    return np.asarray(sorted(set(pick)),int)

# ----------------------- RANSAC 抛物线 -----------------------
def ransac_parabola(s, h, iters=200, min_frac=0.15):
    if len(s)<6: return None, np.zeros_like(s,bool)
    best_inliers=np.zeros_like(s,bool)
    rng=np.random.default_rng(0)
    for _ in range(iters):
        idx=rng.choice(len(s), size=3, replace=False)
        S=np.vstack([s[idx]**2, s[idx], np.ones(3)]).T
        try:
            A,B,C=np.linalg.lstsq(S, h[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pred=A*s*s + B*s + C
        resid=np.abs(h-pred)
        mad=np.median(np.abs(resid-np.median(resid)))+1e-9
        thr=2.5*mad
        inliers=resid<=thr
        if inliers.sum()>best_inliers.sum():
            best_inliers=inliers
    if best_inliers.mean()<min_frac:
        return None, np.zeros_like(s,bool)
    S=np.vstack([s[best_inliers]**2, s[best_inliers], np.ones(best_inliers.sum())]).T
    A,B,C=np.linalg.lstsq(S, h[best_inliers], rcond=None)[0]
    return (A,B,C), best_inliers

# ----------------------- 回收（中心线） -----------------------
def recover_points_by_centerline(xyz_w, assigned_idx_set, a,b, spacing, base_width, width_scale, pr, extra_margin_factor):
    a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
    base_r=base_width+width_scale*L2
    radius=base_r + extra_margin_factor*spacing
    mask=np.ones(len(xyz_w),bool)
    if assigned_idx_set: mask[list(assigned_idx_set)]=False
    pts=xyz_w[mask]
    if len(pts)==0: return np.array([],int)
    ap=pts-a2; t=ap@u2
    tmask=(t>=0)&(t<=L2); tclip=np.clip(t,0.0,L2)
    closest=a2+np.outer(tclip,u2)
    dist=np.linalg.norm(pts-closest,axis=1)
    pick=np.where(tmask&(dist<=radius))[0]
    if len(pick)==0: return np.array([],int)
    return np.where(mask)[0][pick]

# ----------------------- 细化（+保存模型与半径） -----------------------
def refine_spans_with_ransac(tower_centers, xyz_w, spans, span_to_indices, cfg, dz_to_ground):
    if not spans: return spans, span_to_indices, {}, {}
    out_map={}; out_models={}; out_r={}
    for key in spans:
        i,j=key; a,b=tower_centers[i], tower_centers[j]
        pr = CONFIG['protection_radius']
        mem = span_to_indices[key]
        if len(mem)<8: continue
        sub=xyz_w[mem]

        # 自适应基础半径（法向 MAD -> std）
        _, s, L2, u2, a2 = (*centerline_dist(sub,a,b,pr),)
        _, n, _ = span_local_axes(a,b)
        h = (sub - a2) @ n
        mad = np.median(np.abs(h - np.median(h))) + 1e-9
        est_std = 1.4826 * mad
        spacing_local = estimate_point_spacing(sub)
        base_floor = max(2.0*spacing_local, 0.5)
        radius_est = max(base_floor, 2.5*est_std + 0.4)

        # 若段中点贴地，放宽
        mids = a*(1-np.linspace(0,1,5))[:,None] + b*np.linspace(0,1,5)[:,None]
        if np.any(dz_to_ground(mids) < CONFIG.get("ground_relax_dz",3.0)):
            radius_est *= CONFIG.get("relax_width_gain",1.5)

        layers=split_parallel_layers(sub, a,b, max_layers=cfg.get("max_layers",6))
        keep_final=[]; models=[]
        for idx in layers:
            pts=sub[idx]
            s1,h1,L2,_,_,_ = project_to_span(pts, a,b, pr)
            model,inliers = ransac_parabola(s1,h1,
                                            iters=cfg.get("ransac_iters",200),
                                            min_frac=cfg.get("ransac_min_frac",0.15))
            if model is None:
                keep_final.extend(idx.tolist());
                continue
            A,B,C = model
            pred=A*s1*s1 + B*s1 + C
            resid=np.abs(h1-pred)
            mad=np.median(np.abs(resid-np.median(resid)))+1e-9
            band = cfg.get("ransac_band_sigma",2.5)*mad + 0.5
            keep = idx[np.where(resid<=band)[0]]
            keep_final.extend(keep.tolist())
            models.append({"ABC":(A,B,C), "band":band})
        if keep_final:
            kept = mem[np.array(sorted(set(keep_final)),int)]
            out_map[key]=kept
            out_models[key]={"a":a, "b":b, "pr":pr, "models":models}
            out_r[key]=radius_est
    return list(out_map.keys()), out_map, out_models, out_r

# ----------------------- 主细化（回收+桥接+分层清洗） -----------------------
def pca_refine_and_recover_spans(tower_centers, xyz_w, spans, span_to_indices, cfg, aux):
    if not spans: return spans, span_to_indices
    spacing=aux.get("spacing",1.0)
    base_width=aux.get("base_width",1.0)
    w0=aux.get("width_scale",0.02)
    width_scale=max(cfg.get("recover_width_scale",0.03), w0)
    base_enlarge=cfg.get("recover_base_enlarge",1.25)
    extra=cfg.get("pca_refine_extra_margin",2.0)
    base_min=cfg['span_validation_min_points']
    factor=cfg.get("min_points_per_meter_factor",0.25)

    global_assigned=set()
    for idx in span_to_indices.values(): global_assigned.update(idx.tolist())

    refined={}
    for key in spans:
        i,j=key; a,b=tower_centers[i], tower_centers[j]
        members=set(span_to_indices[key].tolist())

        new_idx=recover_points_by_centerline(
            xyz_w, global_assigned, a,b, spacing,
            base_width*base_enlarge, width_scale,
            cfg['protection_radius'], extra
        )
        if len(new_idx)>0: members.update(new_idx.tolist())

        Lspan=np.linalg.norm(b-a)
        min_pts=adaptive_min_points_for_span(Lspan, spacing, base_min, factor)
        if len(members)<min_pts: continue

        mem_arr=np.asarray(sorted(members),int)
        add=bridge_gaps_along_centerline(
            xyz_w, a,b, cfg['protection_radius'],
            mem_arr, spacing,
            max_gap_mult=cfg.get("gap_max_mult",8.0),
            search_radius_mult=cfg.get("gap_search_r_mult",3.0)
        )
        if len(add)>0:
            members.update(add.tolist()); mem_arr=np.asarray(sorted(members),int)

        # 层内 MAD 清洗
        sub=xyz_w[mem_arr]; layers=split_parallel_layers(sub,a,b,max_layers=cfg.get("max_layers",6))
        keep=[]
        u,(n),_ = span_local_axes(a,b)
        for idx in layers:
            h=((sub[idx]-a)@n)
            med=np.median(h); mad=np.median(np.abs(h-med))+1e-6
            keep.extend(idx[np.where(np.abs(h-med)<=3.0*mad)[0]].tolist())
        mem_arr=mem_arr[np.array(sorted(set(keep)),int)]

        mem_arr=filter_by_linearity(xyz_w, mem_arr,
                                    L_min=CONFIG.get("linearity_L_min",0.55),
                                    k=CONFIG.get("linearity_k",12))
        if len(mem_arr)>=min_pts: refined[key]=mem_arr
    return list(refined.keys()), refined

# ----------------------- s 轴 close 辅助 -----------------------
def close_gaps_on_s_axis(xyz_w, key, idx_arr, a,b, pr, spacing, close_gap_mult, close_search_mult, radius_span):
    if len(idx_arr)<5: return np.array([],int)
    a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
    P=xyz_w[idx_arr]
    s=(P-a2)@u2
    s_sorted=np.sort(s)
    small=[]
    thr = close_gap_mult*spacing
    for i in range(len(s_sorted)-1):
        gap = s_sorted[i+1]-s_sorted[i]
        if 0 < gap <= thr:
            small.append((s_sorted[i], s_sorted[i+1]))
    if not small: return np.array([],int)
    return np.array(small, dtype=float)

# ----------------------- 两阶段吸附 + close -----------------------
def snap_stage(points, spans, span_map, span_models, span_radius, spacing, cfg,
               center_r_mult, band_sigma, abs_floor, residual_weight,
               do_close=False, close_gap_mult=6.0, close_search_mult=2.5):
    if not spans: return span_map
    assigned=set()
    for idx in span_map.values(): assigned.update(idx.tolist())
    unassigned = np.array(sorted(set(range(len(points))) - assigned), dtype=int)
    if len(unassigned)==0: return span_map

    cand_best={}  # p -> (score, key)

    for key in spans:
        if key not in span_models:
            continue
        a=span_models[key]["a"]; b=span_models[key]["b"]; pr=span_models[key]["pr"]
        models=span_models[key]["models"]
        if len(models)==0:
            continue
        rad = span_radius.get(key, (center_r_mult*spacing))
        a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
        P = points[unassigned]
        ap = P - a2
        s = ap @ u2
        tmask = (s>=0)&(s<=L2)
        closest=a2+np.outer(np.clip(s,0.0,L2),u2)
        d = np.linalg.norm(P-closest, axis=1)
        near = np.where(tmask & (d <= center_r_mult*spacing + rad))[0]
        if len(near)==0:
            continue

        sel_idx = unassigned[near]
        Q = P[near]; sQ=s[near]; dQ=d[near]
        _,n,_=span_local_axes(a,b)
        h = (Q - a2) @ n

        best_resid = np.full(len(Q), np.inf)
        best_band  = np.full(len(Q), np.inf)
        for mdl in models:
            A,B,C = mdl["ABC"]; band_base = mdl["band"]
            pred = A*(sQ**2) + B*sQ + C
            resid = np.abs(h - pred)
            band = max(band_sigma * (band_base - 0.5), abs_floor)
            better = resid < best_resid
            best_resid[better] = resid[better]
            best_band[better]  = band

        ok = np.where(best_resid <= best_band)[0]
        if len(ok)==0: continue

        score = (dQ[ok]/(rad+1e-9)) + residual_weight*(best_resid[ok]/(best_band[ok]+1e-9))
        for j, gidx in enumerate(sel_idx[ok]):
            sc = float(score[j])
            prev = cand_best.get(int(gidx))
            if (prev is None) or (sc < prev[0]):
                cand_best[int(gidx)] = (sc, key)

        if do_close and key in span_map:
            gaps = close_gaps_on_s_axis(points, key, span_map[key], a,b, pr, spacing,
                                        close_gap_mult, close_search_mult, rad)
            if gaps.size>0:
                sU = (points[unassigned]-a2)@u2
                dU = np.linalg.norm(points[unassigned] - (a2+np.outer(np.clip(sU,0.0,L2),u2)), axis=1)
                sr = close_search_mult*spacing + 0.5*rad
                for (s0,s1) in gaps:
                    m = (sU>=s0) & (sU<=s1) & (dU<=sr)
                    if not np.any(m): continue
                    cand = unassigned[np.where(m)[0]]
                    for gidx in cand:
                        prev = cand_best.get(int(gidx))
                        bonus = 0.05
                        if prev is not None:
                            cand_best[int(gidx)] = (prev[0]-bonus, key)
                        else:
                            cand_best[int(gidx)] = (0.9, key)

    if not cand_best:
        return span_map

    by_key={}
    for p,(sc,key) in cand_best.items():
        by_key.setdefault(key, []).append(p)
    for key,lst in by_key.items():
        arr=np.array(lst,int)
        if key in span_map:
            span_map[key]=np.array(sorted(set(np.concatenate([span_map[key],arr]))),int)
        else:
            span_map[key]=np.array(sorted(set(arr)),int)
    return span_map

# ----------------------- 邻接图（只在邻居间防串色） -----------------------
def build_span_adjacency(spans, tower_centers, cfg):
    """邻接：共享电塔 或 两端点有一对距离 < neighbor_end_dist"""
    neighbor = defaultdict(set)
    if not spans: return neighbor
    # 共享电塔
    by_tower = defaultdict(list)
    for key in spans:
        i,j = key
        by_tower[i].append(key)
        by_tower[j].append(key)
    for _, lst in by_tower.items():
        for a in lst:
            for b in lst:
                if a!=b:
                    neighbor[a].add(b)

    # 端点接近
    nd = cfg.get("neighbor_end_dist", 50.0)
    for a in spans:
        ai,aj = a
        a0, a1 = tower_centers[ai], tower_centers[aj]
        for b in spans:
            if a==b: continue
            bi,bj = b
            b0, b1 = tower_centers[bi], tower_centers[bj]
            dmin = min(np.linalg.norm(a0-b0), np.linalg.norm(a0-b1),
                       np.linalg.norm(a1-b0), np.linalg.norm(a1-b1))
            if dmin <= nd:
                neighbor[a].add(b)
    return neighbor

# ----------------------- 快速防串色（矢量化） -----------------------
def anti_bleed_reassign_fast(points, spans, span_map, span_models, span_radius, spacing, cfg):
    if not spans: return span_map

    w = cfg.get("snap_residual_weight",0.7)
    margin = cfg.get("anti_bleed_margin",0.2)
    band_sigma = cfg.get("snap2_band_sigma",3.8)
    abs_floor  = cfg.get("snap_abs_floor",0.6)

    # 预缓存：每条跨段的几何
    cache = {}
    for key in spans:
        mdl = span_models.get(key)
        if not mdl or len(mdl["models"])==0:
            continue
        a = mdl["a"]; b = mdl["b"]; pr = mdl["pr"]; models = mdl["models"]
        a2,b2,u2,L2 = (*centerline_window(a,b,pr),)
        _, n, _ = span_local_axes(a,b)
        rad = span_radius.get(key, spacing*2.0)
        # 每层最终带宽（用于评分）
        bands = []
        ABCs  = []
        for m in models:
            A,B,C = m["ABC"]; band_base=m["band"]
            band = max(band_sigma*(band_base - 0.5), abs_floor)
            bands.append(band); ABCs.append((A,B,C))
        cache[key] = {"a2":a2, "u2":u2, "L2":L2, "n":n, "rad":rad,
                      "ABCs":np.array(ABCs,float), "bands":np.array(bands,float)}

    # 邻接表
    adj = build_span_adjacency(spans, np.array([m["a"] if "a" in m else [0,0,0] for m in span_models.values()]), cfg)
    # 上面一行只是为了结构对齐；实际几何已在 cache 里

    # 逐跨段批量处理其点，和邻居比较
    moves_by_from = defaultdict(list)  # from_key -> list of (point_idx, to_key)
    for key in spans:
        if key not in span_map or key not in cache:
            continue
        pts_idx = span_map[key]
        if len(pts_idx)==0:
            continue
        dat = cache[key]
        P = points[pts_idx]
        # 当前跨段分数（向量）
        ap = P - dat["a2"]
        s  = ap @ dat["u2"]
        inwin = (s>=0) & (s<=dat["L2"])
        s = s[inwin]; sel_idx = pts_idx[inwin]; Q = P[inwin]
        if len(sel_idx)==0:
            continue
        closest = dat["a2"] + np.outer(np.clip(s,0.0,dat["L2"]), dat["u2"])
        d0 = np.linalg.norm(Q - closest, axis=1)
        h  = (Q - dat["a2"]) @ dat["n"]

        # 残差取所有层的最小
        if len(dat["ABCs"])==0:
            continue
        A = dat["ABCs"][:,0][:,None]; B = dat["ABCs"][:,1][:,None]; C = dat["ABCs"][:,2][:,None]
        pred = (A*(s[None,:]**2) + B*s[None,:] + C)
        resid = np.abs(h[None,:] - pred)
        best_resid = np.min(resid, axis=0)
        best_band  = np.min(dat["bands"][:,None], axis=0)  # 所有层带宽里取最小
        cur_score  = (d0/(dat["rad"]+1e-9)) + w*(best_resid/(best_band+1e-9))

        # 和邻居比较
        neighbors = list(adj.get(key, []))
        if not neighbors:
            continue
        best_alt_score = np.full(len(sel_idx), np.inf)
        best_alt_key   = np.array([None]*len(sel_idx), dtype=object)

        for nb in neighbors:
            if nb not in cache:
                continue
            nbdat = cache[nb]
            # 对同一批点一次性计算分数
            ap2 = Q - nbdat["a2"]
            s2  = ap2 @ nbdat["u2"]
            inwin2 = (s2>=0) & (s2<=nbdat["L2"])
            if not np.any(inwin2):
                continue
            # 只更新窗口内的点
            idx = np.where(inwin2)[0]
            s2v = s2[idx]
            closest2 = nbdat["a2"] + np.outer(np.clip(s2v,0.0,nbdat["L2"]), nbdat["u2"])
            d2 = np.linalg.norm(Q[idx] - closest2, axis=1)
            h2 = (Q[idx] - nbdat["a2"]) @ nbdat["n"]

            if len(nbdat["ABCs"])==0:
                continue
            A2 = nbdat["ABCs"][:,0][:,None]; B2 = nbdat["ABCs"][:,1][:,None]; C2 = nbdat["ABCs"][:,2][:,None]
            pred2 = (A2*(s2v[None,:]**2) + B2*s2v[None,:] + C2)
            resid2 = np.abs(h2[None,:] - pred2)
            best_resid2 = np.min(resid2, axis=0)
            best_band2  = np.min(nbdat["bands"][:,None], axis=0)
            score2 = (d2/(nbdat["rad"]+1e-9)) + w*(best_resid2/(best_band2+1e-9))

            # 若更好，记录
            better = score2 < best_alt_score[idx]
            best_alt_score[idx[better]] = score2[better]
            # 关键修复：将元组 nb 扩展为与左侧切片等长的对象列表，避免广播失败
            cnt = int(np.count_nonzero(better))
            if cnt > 0:
                best_alt_key[idx[better]] = [nb] * cnt

        # 判定是否移动：分数需明显更小
        move_mask = (best_alt_score < (1.0 - margin)*cur_score)
        if np.any(move_mask):
            for pidx, to_key in zip(sel_idx[move_mask], best_alt_key[move_mask]):
                if to_key is not None:
                    moves_by_from[key].append((int(pidx), to_key))

    # 执行移动
    for from_key, items in moves_by_from.items():
        if len(items)==0:
            continue
        keep = set(span_map[from_key].tolist())
        for p, to_key in items:
            if p in keep:
                keep.remove(p)
                arr = span_map.get(to_key, np.array([], int))
                span_map[to_key] = np.array(sorted(set(np.append(arr, p))), int)
        span_map[from_key] = np.array(sorted(keep), int)

    return span_map

# ----------------------- 主流程 -----------------------
def main(input_path, output_path):
    cfg = CONFIG
    print(f"正在加载文件: {input_path}")
    print("模式: V7.1 (快速防串色)")

    xyz_all, cls = load_cloud(input_path)
    if len(xyz_all)==0:
        print("错误：空点云"); return

    ground_mask = (cls==0)
    non_ground_mask = (cls==1)
    xyz_ground = xyz_all[ground_mask]
    xyz_ng = xyz_all[non_ground_mask]
    print(f"地面 {len(xyz_ground)} | 非地面 {len(xyz_ng)}")

    ground_z, dz_to_ground = build_ground_model(xyz_ground)

    print("1) 电塔粗分...")
    mt = tower_mask(xyz_ng, cfg["tower_radius"], cfg["tower_density"], cfg["tower_hmin"], cfg["tower_expand"])
    xyz_t = xyz_ng[mt]
    xyz_w = xyz_ng[~mt]
    print(f"塔点 {len(xyz_t)} | 线/其他 {len(xyz_w)}")
    if len(xyz_t)==0:
        print("未识别电塔"); return

    print(" -> 非塔去噪")
    xyz_w = remove_outliers(xyz_w, cfg["noise_nb"], cfg["noise_r"])

    print("2) 电塔聚类...")
    centers, tower_labels = get_tower_centers(xyz_t, cfg)
    print(f"电塔数: {len(centers)}")
    if len(centers)<2: print("电塔不足2"); return

    # 生成tower.json（从process_2.py借用）
    print(" -> 正在生成 tower.json...")
    tower_info_list = []
    if tower_labels is not None and len(tower_labels) > 0:
        for i in range(len(centers)):
            cluster_points = xyz_t[tower_labels == i]
            if len(cluster_points) > 0:
                tower_height = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
                point_count = len(cluster_points)

                tower_info_list.append({
                    "tower_id": i + 1,
                    "point_count": int(point_count),
                    "height_m": float(f"{tower_height:.2f}"),
                    "center_x": float(f"{centers[i][0]:.2f}"),
                    "center_y": float(f"{centers[i][1]:.2f}"),
                    "center_z": float(f"{centers[i][2]:.2f}")
                })

    tower_json_path = os.path.join(os.path.dirname(output_path), 'tower.json')
    with open(tower_json_path, 'w', encoding='utf-8') as f:
        json.dump(tower_info_list, f, indent=4, ensure_ascii=False)
    print(f" -> 已保存电塔信息到: {tower_json_path}")

    print("3) 候选跨段发现 + 地面感知 + 冲突消解 ...")
    spans, span2idx, aux = discover_and_assign_spans_adaptive(centers, xyz_w, cfg, dz_to_ground)
    print(f"初筛跨段: {len(spans)}")
    if not spans: print("未找到跨段"); return

    print("4) 回收 + 桥接 + 分层清洗 ...")
    spans, span2idx = pca_refine_and_recover_spans(centers, xyz_w, spans, span2idx, cfg, aux)
    print(f"细化后跨段: {len(spans)}")

    print("5) RANSAC 拟合并保存模型 + 每跨自适应半径 ...")
    spans, span2idx, span_models, span_radius = refine_spans_with_ransac(centers, xyz_w, spans, span2idx, cfg, dz_to_ground)
    print(f"RANSAC 后跨段: {len(spans)}")

    spacing = aux.get("spacing", estimate_point_spacing(xyz_w))

    print("6) 吸附阶段一（保守）...")
    span2idx = snap_stage(
        xyz_w, spans, span2idx, span_models, span_radius, spacing, cfg,
        center_r_mult=cfg.get("snap1_center_r_mult",1.8),
        band_sigma=cfg.get("snap1_band_sigma",2.8),
        abs_floor=cfg.get("snap_abs_floor",0.6),
        residual_weight=cfg.get("snap_residual_weight",0.7),
        do_close=False
    )

    print("7) 吸附阶段二（进取 + s轴 close）...")
    span2idx = snap_stage(
        xyz_w, spans, span2idx, span_models, span_radius, spacing, cfg,
        center_r_mult=cfg.get("snap2_center_r_mult",2.8),
        band_sigma=cfg.get("snap2_band_sigma",3.8),
        abs_floor=cfg.get("snap_abs_floor",0.6),
        residual_weight=cfg.get("snap_residual_weight",0.7),
        do_close=True,
        close_gap_mult=cfg.get("close_gap_mult",6.0),
        close_search_mult=cfg.get("close_search_mult",2.5)
    )

    print("8) 防串色（邻居矢量化） ...")
    span2idx = anti_bleed_reassign_fast(xyz_w, spans, span2idx, span_models, span_radius, spacing, cfg)

    # 写标签
    span_labels = np.zeros(len(xyz_w), int)
    for sid,key in enumerate(spans, start=1):
        idx = span2idx.get(key, np.array([], int))
        if len(idx)>0:
            span_labels[idx]=sid

    # 使用process_3.py的分类逻辑和颜色方案
    print("9) 导出 LAS (采用 process_3 风格)...")
    # 定义调色板和固定颜色（从process_3.py借用）
    palette = np.array([
        [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
        [31, 119, 180], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207]
    ], dtype=np.uint8)
    ground_color = np.array([150, 150, 150], dtype=np.uint8)   # 深灰色
    tower_color = np.array([173, 216, 230], dtype=np.uint8)  # 浅蓝色
    default_color = np.array([128, 128, 128], dtype=np.uint8) # 灰色

    # 分离出不同类别的点云数据
    processed_wire_mask = span_labels > 0
    unprocessed_wire_mask = span_labels == 0

    xyz_wire_ok = xyz_w[processed_wire_mask]
    xyz_wire_noise = xyz_w[unprocessed_wire_mask]

    # 为每个类别创建对应的属性数组
    # -- 颜色 --
    rgb_ground = np.tile(ground_color, (len(xyz_ground), 1))
    rgb_tower = np.tile(tower_color, (len(xyz_t), 1))
    rgb_wire_noise = np.tile(default_color, (len(xyz_wire_noise), 1))
    # 为成功分类的导线根据ID应用调色板
    labels_ok = span_labels[processed_wire_mask] - 1  # 转换为0-based索引
    rgb_wire_ok = palette[labels_ok % len(palette)]

    # -- 分类 (Classification) --
    # 采用标准分类值: 2=地面, 6=建筑(电塔), 14=导线, 1=未分类
    cls_ground = np.full(len(xyz_ground), 2, dtype=np.uint8)
    cls_tower = np.full(len(xyz_t), 6, dtype=np.uint8)
    cls_wire_ok = np.full(len(xyz_wire_ok), 14, dtype=np.uint8)
    cls_wire_noise = np.full(len(xyz_wire_noise), 1, dtype=np.uint8)

    # -- 线路ID (line_id) --
    lid_ground = np.zeros(len(xyz_ground), dtype=np.uint16)
    lid_tower = np.zeros(len(xyz_t), dtype=np.uint16)
    lid_wire_ok = span_labels[processed_wire_mask].astype(np.uint16)  # ID从1开始
    lid_wire_noise = np.zeros(len(xyz_wire_noise), dtype=np.uint16)

    # -- 段ID (process_id) - 根据跨段分配process值
    pid_ground = np.zeros(len(xyz_ground), dtype=np.uint16)
    pid_tower = np.ones(len(xyz_t), dtype=np.uint16)

    # 为导线分配process_id，从2开始
    process_w = np.ones(len(xyz_w), dtype=np.uint16)
    uniq = np.unique(span_labels[span_labels > 0])
    for i, sid in enumerate(uniq):
        process_w[span_labels == sid] = i + 2

    pid_wire_ok = process_w[processed_wire_mask]
    pid_wire_noise = process_w[unprocessed_wire_mask]

    # 按相同顺序安全地堆叠所有数据
    # 顺序: 地面 -> 电塔 -> 已分类导线 -> 噪声导线
    final_xyz = np.vstack((xyz_ground, xyz_t, xyz_wire_ok, xyz_wire_noise))
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
    final_wire_count = len(uniq)
    print(f"* 保存 {output_path} | 跨段 {final_wire_count} | 地面点 {len(xyz_ground)}")
    print(f"Process: 地面=0, 电塔=1, 跨段=2..{1+final_wire_count}")

# ------------------------------------------------------------------
# 脚本入口：解析命令行参数
# ------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python debug3.py <input_file_path> <output_file_path>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"错误: 输入文件不存在 -> {in_path}")
        sys.exit(1)

    main(in_path, out_path)
