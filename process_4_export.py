# -*- coding: utf-8 -*-
import laspy
import numpy as np
import json
import sys
import os
import random

# 定义地面点的安全阈值，超过此数量将被抽样
GROUND_POINT_THRESHOLD = 2500000

def export_for_webgl(input_path, output_path):
    """
    读取由 process_3.py 生成的 3.las 文件，并为三维重建导出优化后的 JSON 数据。
    - 使用 ASPRS 标准分类代码进行精确识别。
    - 完整保留电力塔 (classification=6) 和其他非地面/非导线点。
    - 仅对地面点 (classification=2) 在数量过多时进行抽样。
    - 提取每条电力线 (classification=14) 的点及其已分配好的颜色。
    """
    print(f"--- 正在为 WebGL 准备数据 (最终版): {input_path} ---")
    try:
        las = laspy.read(input_path)

        # 确认所有必需的属性都存在
        required_attrs = ['x', 'y', 'z', 'red', 'green', 'blue', 'classification']
        if not all(hasattr(las, attr) for attr in required_attrs):
            print("错误: .las 文件缺少必要的属性 (坐标, 颜色, 分类)。")
            return False

        # 提取核心数据
        xyz = np.vstack((las.x, las.y, las.z)).transpose()
        rgb = np.vstack((las.red >> 8, las.green >> 8, las.blue >> 8)).transpose().astype(np.uint8)
        classification = las.classification

        # --- 1. 根据 process_3.py 的输出，使用标准分类代码创建掩码 ---
        # 14 = Wire - Conductor
        # 6  = Building (我们在流程中用它代表电力塔)
        # 2  = Ground
        powerline_mask = classification == 14
        tower_mask = classification == 6
        ground_mask = classification == 2
        
        # “其余点”是所有不属于上述三类的点 (例如 process_3 未成功分类的噪声点)
        remaining_mask = ~(powerline_mask | tower_mask | ground_mask)

        # --- 2. 分别处理每个类别 ---

        # a) 电力塔点 (全部保留)
        tower_points_xyz = xyz[tower_mask]
        tower_points_rgb = rgb[tower_mask]
        print(f" -> 找到并完整保留 {len(tower_points_xyz)} 个电力塔点 (分类=6)。")

        # b) 地面点 (按需抽样)
        ground_points_xyz = xyz[ground_mask]
        ground_points_rgb = rgb[ground_mask]
        num_ground_points = len(ground_points_xyz)
        print(f" -> 找到 {num_ground_points} 个地面点 (分类=2)。")

        if num_ground_points > GROUND_POINT_THRESHOLD:
            print(f"警告: 地面点数量超过阈值 {GROUND_POINT_THRESHOLD}，将进行随机抽样...")
            indices_to_keep = random.sample(range(num_ground_points), GROUND_POINT_THRESHOLD)
            ground_points_xyz = ground_points_xyz[indices_to_keep]
            ground_points_rgb = ground_points_rgb[indices_to_keep]
            print(f" -> 抽样后保留 {len(ground_points_xyz)} 个地面点。")

        # c) 其余点 (全部保留)
        remaining_points_xyz = xyz[remaining_mask]
        remaining_points_rgb = rgb[remaining_mask]
        print(f" -> 找到并完整保留 {len(remaining_points_xyz)} 个其余点。")

        # d) 合并所有非电力线点
        final_other_xyz = np.vstack((tower_points_xyz, ground_points_xyz, remaining_points_xyz))
        final_other_rgb = np.vstack((tower_points_rgb, ground_points_rgb, remaining_points_rgb))
        other_points_data = np.hstack((final_other_xyz, final_other_rgb)).tolist()
        print(f" -> 最终导出的'other_points'总数为: {len(other_points_data)}")

        # --- 3. 处理电力线 (提取点和颜色) ---
        powerlines_grouped = {}
        if 'line_id' in las.point_format.dimension_names and np.any(powerline_mask):
            powerline_xyz = xyz[powerline_mask]
            powerline_rgb = rgb[powerline_mask]
            powerline_ids = las.line_id[powerline_mask]
            
            unique_ids = np.unique(powerline_ids)

            for line_id in unique_ids:
                if line_id == 0: continue
                
                line_mask = powerline_ids == line_id
                line_points = powerline_xyz[line_mask]
                
                # 获取该线的第一个点的颜色作为代表色
                # 因为 process_3.py 已经为同一根线的所有点设置了相同颜色
                line_color = powerline_rgb[line_mask][0].tolist()

                # 按X坐标排序，以确保曲线路径的顺序基本正确
                sorted_indices = np.argsort(line_points[:, 0])
                sorted_line_points = line_points[sorted_indices]

                powerlines_grouped[str(line_id)] = {
                    "points": sorted_line_points.tolist(),
                    "color": line_color  # 将颜色信息加入JSON
                }
            print(f" -> 找到并分组了 {len(powerlines_grouped)} 根独立的电力线。")
        else:
            print(" -> 未找到 'line_id' 字段或分类为14的电力线点。")

        # --- 4. 整合并导出最终的JSON对象 ---
        output_data = {
            "other_points": other_points_data,
            "powerlines": powerlines_grouped
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f)

        print(f"--- WebGL 数据已成功导出到: {output_path} ---")
        return True

    except Exception as e:
        print(f"导出 WebGL 数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法: python process_4_export.py <input_3.las> <output_webgl_data.json>")
        sys.exit(1)

    input_file, output_file = sys.argv[1], sys.argv[2]
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在 -> {input_file}")
        sys.exit(1)

    export_for_webgl(input_file, output_file)