import numpy as np
import triangle as tr
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
import gc
from REFPROP_p import REFPROP

params = {
    'fluid': 'PARAHYDROGEN',
    'T_min': 17.0,                  # K
    'T_max': 310.0,                 # K
    'P_min': 2.5e4,                 # Pa
    'P_max': 1.0025e7,              # Pa
    'max_nodes': 1e5,               # 最大节点数
    'max_iter': 9.5e4,              # 最大迭代次数
    'plot_interval': 2500,          # 绘图间隔
    'save_file': 'triprp_data.npz', # 保存数据的文件名
    'criterion': 'CP',              # 误差评估的属性类型
    'properties': ['CP', 'D', 'VIS', 'TCX']
}

tolerance = {
    'CP': 1e-3,
    'D': 1e-3,
    'VIS': 5e-3,
    'TCX': 5e-3
}

# 初始化REFPROP对象
ref = REFPROP(params['fluid'], refprop_path=r'D:\1_UserPrograms\Refprop\REFPROP')

# 获取定常属性
const_prop = {
    'Tc': ref.PRP_CRIT('TC'),           #   临界温度
    'Pc': ref.PRP_CRIT('PC'),           #   临界压力
    'Ttrip': ref.PRP_TRIP('T_trip'),    #   三相点温度
    'Ptrip': ref.PRP_TRIP('P_trip'),    #   三相点压力
}

def colon(start, step, end):
    "生成等差数列"
    return np.arange(start, end + step/2, step)

def get_SAT_data():

    points = []
    P_sat_list = np.concatenate([
        colon(4e4, 1e4, 1e5),
        [101325.0],
        colon(1.1e5, 1e4, 1.2e6),
        colon(1.205e6, 5e3, 1.24e6),
        colon(1.243e6, 3e3, 1.27e6),
        colon(1.271e6, 1e3, 1.284e6),
        colon(1.2841e6, 1e2, 1.2857e6)
    ])

    for P in P_sat_list:
        try:
            T_SAT = ref.PRP_SAT('T', 'P', P, 'liquid')
            points.append((P, T_SAT))
        except Exception as e:
            points.append((P, np.nan))
            continue
    
    points = np.concatenate([
        [(ref.PRP_SAT('P', 'T', params['T_min'], 'liquid'), params['T_min'])],
        points,
        [(const_prop['Pc'], const_prop['Tc'])]
    ])

    points = sorted(points, key=lambda x: x[0])

    return np.array(points)

def get_MELT_data():
    """失败，不可使用"""
    points = []
    P_melt_list = colon(1e4, 2e4, 1e7)
    trip_point = (const_prop['Ptrip'], const_prop['Ttrip'])

    for P in P_melt_list:
        try:
            T_MELT = ref.PRP_MELT('P', P)
            if T_MELT >= params['T_max']:
                try:
                    P_at_max = ref.PRP_MELT('T', params['T_max'])
                except Exception as e:
                    P_at_max = np.nan
                end_point = (P_at_max, params['T_max'])
                points = [trip_point] + points + [end_point]
                points.sort(key=lambda x: x[0])
                return np.array(points)
            
            points.append((P, T_MELT))
        except Exception as e:
            points.append((P, np.nan))
            continue

    points = [trip_point] + points
    points.sort(key=lambda x: x[0])
    return np.array(points)

def get_WIDOM_data():
    def skip_even_rows(row_index):
        # 跳过偶数行
        return (row_index % 2) == 0
    
    file_path = "D:\\11_Projects\\H2TankCFD\\FLuent_Cases\\Properties\\Unstructured_Table\\Triprp\\widom_data_parahydrogen.xlsx"
    df = pd.read_excel(file_path, sheet_name='Sheet1', usecols='A:B', header=None, skiprows=skip_even_rows)
    points = list(zip(df[0], df[1]))

    return np.array(points)

def generate_seed_points(P_min, P_max, T_min, T_max, n_pt=30, n_pt_T=20):
    """
    在矩形区域内生成均匀网格种子点（不排除任何点）。
    """
    P_vals = np.linspace(P_min+1e3, P_max-1e3, n_pt)
    T_vals = np.linspace(T_min+1, T_max-1, n_pt_T)
    grid_P, grid_T = np.meshgrid(P_vals, T_vals)
    grid_points = np.column_stack([grid_P.ravel(), grid_T.ravel()])
    print(f"生成了 {len(grid_points)} 个种子点")
    return grid_points

def barycentric_coords(p, v0, v1, v2):
    """计算点 p 相对于三角形 (v0, v1, v2) 的重心坐标"""
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def evaluate_triangle_error(triangle, points, prp_vals, cache):
    """
    评估三角形triangle内的属性插值误差
    三角形由三个顶点索引组成
    返回最大相对误差
    """
    def tri_sample_points(v0, v1, v2):
        """在三角形内部生成采样点：中心 + 边三等分点"""
        samples = []
        # 重心
        samples.append((v0 + v1 + v2) / 3.0)
        # 边 AB 的两个三等分点
        samples.append(v0 + (v1 - v0) / 3.0)
        samples.append(v0 + 2.0 * (v1 - v0) / 3.0)
        # 边 BC 的两个三等分点
        samples.append(v1 + (v2 - v1) / 3.0)
        samples.append(v1 + 2.0 * (v2 - v1) / 3.0)
        # 边 CA 的两个三等分点
        samples.append(v2 + (v0 - v2) / 3.0)
        samples.append(v2 + 2.0 * (v0 - v2) / 3.0)
        return samples
    
    v0, v1, v2 = points[triangle]
    max_norm_err = 0.0
    samples = tri_sample_points(v0, v1, v2)

    for p in samples:
        key = (p[0], p[1])
        if key not in cache:
            true_vals = {}
            for name in prp_vals.keys():
                try:
                    true_vals[name] = ref.PRP(name, 'T', p[1], 'P', p[0])
                except:
                    true_vals[name] = np.nan
            cache[key] = true_vals
        true_vals = cache[key]

        u, v, w = barycentric_coords(p, v0, v1, v2)
        for name in prp_vals.keys():
            # interp
            vals = prp_vals[name]
            interp = u * vals[triangle[0]] + v * vals[triangle[1]] + w * vals[triangle[2]]
            err = abs(interp - true_vals[name]) / (abs(true_vals[name]) + 1e-12)
            norm_err = err / tolerance[name]
            if norm_err > max_norm_err:
                max_norm_err = norm_err

    return max_norm_err

def get_boundary_points(T_min, T_max, P_min, P_max):
    """生成边界点（按逆时针顺序，闭合）"""
    points = []
    # 下边界
    for t in colon(T_min, 1.0, T_max):
        points.append((P_min, t))
    # 右边界
    for p in colon(P_min + 2.5e4, 2.5e4, P_max):
        points.append((p, T_max))
    # 上边界
    for t in colon(T_max - 1.0, -1.0, T_min):
        points.append((P_max, t))
    # 左边界    
    for p in colon(P_max - 2.5e4, -2.5e4, P_min):
        points.append((p, T_min))

    return np.array(points)

# step1：生成初始三角剖分

# 矩形边界点
rect_points = get_boundary_points(params['T_min'], params['T_max'], params['P_min'], params['P_max'])

# 饱和线数据点
sat_points = get_SAT_data()

# Widom线数据点
widom_points = get_WIDOM_data()

# 生成种子点
seed_points = generate_seed_points(params['P_min'], params['P_max'],
                                   params['T_min'], params['T_max'],
                                   n_pt=80, n_pt_T=60,  # 可根据需要调整密度
)

# 合并所有初始点
init_points = [rect_points, sat_points, widom_points, seed_points]
points = np.vstack(init_points)

# 去重
unique_points, inv  = np.unique(np.round(points, decimals=10), axis=0, return_inverse=True)
points = unique_points

# 原始索引范围
n_rect = len(rect_points)
n_sat = len(sat_points)
n_widom = len(widom_points)

rect_orig_indices = list(range(n_rect))
sat_orig_indices  = list(range(n_rect, n_rect + n_sat))
widom_orig_indices = list(range(n_rect + n_sat, n_rect + n_sat + n_widom))

# 映射到去重后的索引
rect_indices = [inv[i] for i in rect_orig_indices]
sat_indices   = [inv[i] for i in sat_orig_indices]
widom_indices = [inv[i] for i in widom_orig_indices]

print(f"初始点数量: {len(points)}")

# 计算每个初始节点的的prp
prp_names = params['properties']
prp_vals = {name: [] for name in prp_names}
for i, (P, T) in enumerate(points):
    vals = {}
    for name in prp_names:
        try:
            vals[name] = ref.PRP(name, 'T', T, 'P', P)
        except Exception as e:
            print(f"Error calculating PRP for point {i}: P = ({P:.3e}, T = {T:.2f}): {e}")
            vals[name] = np.nan
                
    for name in prp_names:
        prp_vals[name].append(vals[name])

for name in prp_names:
    prp_vals[name] = np.array(prp_vals[name])

# 构建约束
print("正在构建约束...")

# 建立坐标到索引的映射
coord_to_idx = {tuple(p): i for i, p in enumerate(points)}

segments = []

def find_closest_idx(pt):
    dist = np.linalg.norm(points - pt, axis=1)
    return np.argmin(dist)

# 添加矩形边界约束
for i in range(len(rect_points)-1):
    segments.append([rect_indices[i], rect_indices[i+1]])
segments.append([rect_indices[-1], rect_indices[0]])

# 添加饱和线约束
sat_order = np.argsort(sat_points[:, 1])
sat_sorted_indices = [sat_indices[i] for i in sat_order]
for i in range(len(sat_sorted_indices)-1):
    segments.append([sat_sorted_indices[i], sat_sorted_indices[i+1]])

# Widom 线（按压力排序）
widom_order = np.argsort(widom_points[:, 0])
widom_sorted_indices = [widom_indices[i] for i in widom_order]
for i in range(len(widom_sorted_indices)-1):
    segments.append([widom_sorted_indices[i], widom_sorted_indices[i+1]])

# step2: 初始三角剖分
print("正在进行初始三角剖分...")
mesh = tr.triangulate({'vertices': points, 'segments': segments}, 'p')
triangles = mesh['triangles']
print(f"初始三角形数量: {len(triangles)}")

# step3: 迭代细化
print("正在进行迭代细化...")
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
iter_hist, error_hist, nodes_hist = [], [], []

try:
    iter = 0
    while iter < params['max_iter']:
        iter += 1

        # 评估每个三角形的误差
        tri_errors = []
        cache = {}
        for i, tri in enumerate(triangles):
            err = evaluate_triangle_error(tri, points, prp_vals, cache)
            tri_errors.append(err)

        max_norm_error = max(tri_errors)
        iter_hist.append(iter)
        error_hist.append(max_norm_error)
        nodes_hist.append(len(points))

        # 更新监控器
        ax1.clear()
        ax1.semilogy(iter_hist, error_hist, 'b-')  # 对数坐标
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Normalized Max Error')
        ax1.grid(True)

        ax2.clear()
        ax2.plot(iter_hist, nodes_hist, 'r-')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Number of Nodes')
        ax2.grid(True)

        plt.pause(0.01)

        if max_norm_error < 1.0 or len(points) >= params['max_nodes']:
            print("达到终止条件")
            break

        # 最大误差三角形
        worst_idx = np.argmax(tri_errors)
        worst_tri = triangles[worst_idx]

        # 插入新点：重心
        new_point = (points[worst_tri[0]] + points[worst_tri[1]] + points[worst_tri[2]]) / 3.0

        # 添加新点到点集
        points = np.vstack([points, new_point])

        # 计算新的prp
        new_vals = {}
        for name in prp_names:
            try:
                new_vals[name] = ref.PRP(name, 'T', new_point[1], 'P', new_point[0])
            except Exception as e:
                print(f"新点计算失败 ({name}): {e}")
                new_vals[name] = np.nan
    
        for name in prp_names:
            prp_vals[name] = np.append(prp_vals[name], new_vals[name])

        new_segments = []
    
        # 添加矩形边界约束
        for i in range(len(rect_points)-1):
            new_segments.append([rect_indices[i], rect_indices[i+1]])
        new_segments.append([rect_indices[-1], rect_indices[0]])

        # 添加饱和线约束
        sat_order = np.argsort(sat_points[:, 1])
        sat_sorted_indices = [sat_indices[i] for i in sat_order]
        for i in range(len(sat_sorted_indices)-1):
            new_segments.append([sat_sorted_indices[i], sat_sorted_indices[i+1]])

        # Widom 线（按压力排序）
        widom_order = np.argsort(widom_points[:, 0])
        widom_sorted_indices = [widom_indices[i] for i in widom_order]
        for i in range(len(widom_sorted_indices)-1):
            new_segments.append([widom_sorted_indices[i], widom_sorted_indices[i+1]])

        # 重新剖分
        mesh = tr.triangulate({
            'vertices': points,
            'segments': new_segments,
        }, 'p')
        triangles = mesh['triangles']

        # 可视化
        if params['plot_interval'] and iter % params['plot_interval'] == 0:
            plt.figure(figsize=(12, 8))
            plt.triplot(points[:,0], points[:,1], triangles, 'b-', lw=0.3)
            plt.xlabel('Pressure (Pa)')
            plt.ylabel('Temperature (K)')
            plt.title(f'Iteration {iter}, Nodes: {len(points)}')
            plt.grid(True)
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    pass
except KeyboardInterrupt:
    del points
    del triangles
    del prp_vals
    if 'cache' in locals():
        del cache
    import gc
    gc.collect()
    exit(0)

# Save
print("保存结果...")
# 保存物性
node_props = []
for i, (P, T) in enumerate(points):
    try:
        D = ref.PRP('D', 'T', T, 'P', P)
        Cp = ref.PRP('CP', 'T', T, 'P', P)
        V = ref.PRP('VIS', 'T', T, 'P', P)
        K = ref.PRP('TCX', 'T', T, 'P', P)
        H = ref.PRP('H', 'T', T, 'P', P)
        Cv = ref.PRP('CV', 'T', T, 'P', P)
    except Exception as e:
        print(f"节点 {i} 物性计算失败: {e}")
        D = Cp = Cv = V = K = H = np.nan
    node_props.append([D, Cp, V, K, H, Cv])

node_props = np.array(node_props)

np.savez(params['save_file'],
         points=points,
         triangles=triangles,
         props=node_props,
         prop_names=['Density', 'Cp', 'Viscosity', 'Thermal_conductivity', 'Enthalpy', 'Cv'])
print(f"结果已保存到 {params['save_file']}")

# 最终可视化
plt.figure(figsize=(12, 8))
plt.triplot(points[:,0]/1e3, points[:,1], triangles, 'b-', lw=0.3)
plt.xlabel('Pressure (kPa)')
plt.ylabel('Temperature (K)')
plt.title(f'Final mesh, Nodes: {len(points)}')
plt.grid(True)
plt.show()

del points
del triangles
del prp_vals
del cache
del new_vals
del mesh
gc.collect()