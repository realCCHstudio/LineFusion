import open3d as o3d
import numpy as np
import laspy
import pandas as pd
import traceback
import sys          # 新增：支持命令行参数
import os           # 新增：用于检测文件存在

# ==============================================================================
# 0. 参数设置（这里依旧是默认值；真正执行时可被命令行覆盖）
# ==============================================================================
PARAMS = {
    # --- 输入 / 输出文件 ---
    "input_file": "labeled.las",
    "output_file": "powerlines_extracted_0431.las",

    # --- 0a. 统计离群点滤波 (前置噪声移除) ---
    "sor_filter": {
        "nb_neighbors": 15,
        "std_ratio": 2.0
    },

    # --- 1. 归一化高程与地面滤波参数 ---
    "ground_filter": {
        "grid_size": 3.0,
        "percentile": 5,
        "normalized_height_threshold": 20,
    },

    # --- 2. 密度滤波参数 ---
    "density_filter": {
        "radius": 1.0,
        "max_neighbors": 15,
    },

    # --- 3. 几何与聚类滤波参数 ---
    "geometric_filter": {
        "eps": 2.0,
        "min_points": 10,
        "min_cluster_points": 20,
        "linearity_threshold": 0.8,
    }
}

# ==============================================================================
# 辅助函数与核心处理流程（保持不变）
# ==============================================================================

def load_las_file(file_path):
    """使用 laspy 读取 .las 文件并转换为 Open3D 点云对象"""
    print(f"--- 正在加载 LAS 文件: {file_path} ---")
    try:
        las = laspy.read(file_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        print(f"加载成功: {len(pcd.points)} 个点。")
        return pcd, las
    except Exception as e:
        print(f"加载 LAS 文件时出错: {e}")
        return None, None

def save_colored_and_classified_las(file_path, original_las, power_line_indices):
    """
根据索引将原始 LAS 中的电力线点标记为红色且分类=1，其余分类=0，然后保存
    """
    if power_line_indices.size == 0:
        print("警告: 没有识别出电力线点，不保存文件。")
        return

    print(f"--- 正在标记电力线并保存到: {file_path} ---")
    try:
        has_rgb = all(d in original_las.point_format.dimension_names for d in ('red', 'green', 'blue'))
        out_pf_id = original_las.header.point_format.id if has_rgb else 3
        if not has_rgb:
            print("警告: 原始文件不含 RGB，将升级到点格式 3 以写入颜色。")

        new_las = laspy.create(point_format=out_pf_id,
                               file_version=original_las.header.version)
        new_las.header.offsets = original_las.header.offsets
        new_las.header.scales = original_las.header.scales
        new_las.points = original_las.points.copy()

        # 分类
        print("正在设置自定义分类代码...")
        new_las.classification[:] = 0
        new_las.classification[power_line_indices] = 1
        print(f"{power_line_indices.size} 个电力线点已分类为 1。")

        # 颜色
        if not has_rgb:
            new_las.red[:] = 32768
            new_las.green[:] = 32768
            new_las.blue[:] = 32768

        print(f"正在将 {power_line_indices.size} 个电力线点标记为红色。")
        new_las.red[power_line_indices] = 65535
        new_las.green[power_line_indices] = 0
        new_las.blue[power_line_indices] = 0

        new_las.write(file_path)
        print("文件已成功保存。")
    except Exception as e:
        print(f"保存处理后的 LAS 文件时出错: {e}")
        traceback.print_exc()

def filter_statistical_outliers(pcd, nb_neighbors, std_ratio):
    """统计离群点移除 (SOR)"""
    print("--- 0a. 正在进行统计离群点滤波 (SOR) ---")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)
    print(f"SOR 滤波完成: 移除了 {len(pcd.points) - len(cl.points)} 个离群点。")
    return np.asarray(ind)

def calculate_normalized_height_robust(points, grid_size, percentile):
    """稳健归一化高程 (使用百分位数抵抗噪声)"""
    print(f"--- 1a. 正在计算归一化高程 (百分位数={percentile}) ---")
    min_bound = np.min(points, axis=0)
    gx = ((points[:, 0] - min_bound[0]) // grid_size).astype(int)
    gy = ((points[:, 1] - min_bound[1]) // grid_size).astype(int)

    df = pd.DataFrame({'gx': gx, 'gy': gy, 'z': points[:, 2]})
    ground_map = df.groupby(['gx', 'gy'])['z'].apply(lambda zs: np.percentile(zs, percentile))

    df['grid_id'] = list(zip(df.gx, df.gy))
    df['ground_z'] = df['grid_id'].map(ground_map)
    df['ground_z'].fillna(np.min(points[:, 2]), inplace=True)

    print("归一化高程计算完成。")
    return (df['z'] - df['ground_z']).values

def select_points_by_index(pcd, indices):
    """手动 select_by_index，兼容旧版 open3d"""
    pts = np.asarray(pcd.points)[indices]
    np_pcd = o3d.geometry.PointCloud()
    np_pcd.points = o3d.utility.Vector3dVector(pts)
    return np_pcd

def filter_by_density(pcd, radius, max_neighbors):
    """密度滤波：提取稀疏点"""
    print("--- 2. 正在进行密度滤波 ---")
    kdt = o3d.geometry.KDTreeFlann(pcd)
    pts = np.asarray(pcd.points)
    sparse_idx = [i for i, p in enumerate(pts)
                  if len(kdt.search_radius_vector_3d(p, radius)[1]) < max_neighbors]
    return np.array(sparse_idx, dtype=int)

def extract_linear_clusters(pcd, eps, min_points, min_cluster_points, linearity_threshold):
    """DBSCAN + PCA 提取线性簇"""
    print("--- 3. 正在进行聚类与几何滤波 ---")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points,
                                         print_progress=False))
    max_label = labels.max()
    if max_label == -1:
        print("DBSCAN 未形成任何簇。")
        return np.array([], dtype=int)

    print(f"DBSCAN 聚类完成，共 {max_label + 1} 个簇。")
    linear_idx = []
    for i in range(max_label + 1):
        ci = np.where(labels == i)[0]
        if len(ci) < min_cluster_points:
            continue
        cp = np.asarray(select_points_by_index(pcd, ci).points)
        if len(cp) < 2 or np.all(cp == cp[0]):
            continue
        cov = np.cov(cp, rowvar=False)
        if cov.shape != (3, 3) or np.isnan(cov).any():
            continue
        try:
            eigvals, _ = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[0] > 1e-9:
            linearity = (eigvals[0] - eigvals[1]) / eigvals[0]
            if linearity > linearity_threshold:
                linear_idx.append(ci)
    return np.concatenate(linear_idx) if linear_idx else np.array([], dtype=int)

# ==============================================================================
# 主函数：改为接收文件路径参数
# ==============================================================================
def main(input_path, output_path):
    """主执行流程"""
    pcd_orig, las_orig = load_las_file(input_path)
    if pcd_orig is None:
        return

    # 0a. SOR 离群点滤波
    inlier_idx = filter_statistical_outliers(pcd_orig, **PARAMS["sor_filter"])
    pcd_inliers = select_points_by_index(pcd_orig, inlier_idx)
    if not pcd_inliers.has_points():
        print("离群点滤波后无剩余点，终止。")
        return

    # 1. 地面滤波 (稳健)
    inlier_pts = np.asarray(pcd_inliers.points)
    norm_h = calculate_normalized_height_robust(
        inlier_pts,
        PARAMS["ground_filter"]["grid_size"],
        PARAMS["ground_filter"]["percentile"]
    )
    non_ground_rel = np.where(norm_h > PARAMS["ground_filter"]["normalized_height_threshold"])[0]
    non_ground_abs = inlier_idx[non_ground_rel]
    non_ground_pcd = select_points_by_index(pcd_orig, non_ground_abs)
    print(f"地面滤波完成，非地面点数: {len(non_ground_pcd.points)}")
    if not non_ground_pcd.has_points():
        return

    # 2. 密度滤波
    sparse_rel = filter_by_density(non_ground_pcd, **PARAMS["density_filter"])
    sparse_abs = non_ground_abs[sparse_rel]
    sparse_pcd = select_points_by_index(pcd_orig, sparse_abs)
    print(f"密度滤波完成，稀疏点数: {len(sparse_pcd.points)}")
    if not sparse_pcd.has_points():
        return

    # 3. 几何滤波
    power_rel = extract_linear_clusters(sparse_pcd, **PARAMS["geometric_filter"])
    power_abs = sparse_abs[power_rel]
    print(f"几何滤波完成，识别电力线点: {power_abs.size}")

    # 4. 保存
    save_colored_and_classified_las(output_path, las_orig, power_abs)
    print("\n--- 处理完成 ---")

# ==============================================================================
# 脚本入口：解析命令行参数
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("错误: 参数数量不正确。\n"
              "用法: python las_process.py <input_file_path> <output_file_path>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(in_path):
        print(f"错误: 输入文件不存在 -> {in_path}")
        sys.exit(1)

    main(in_path, out_path)
