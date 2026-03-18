# 透视图生成全景图功能需求

## 概述

基于单张透视图（perspective image）和文本 prompt，利用扩散模型逐步生成完整 360° 全景图（equirectangular projection, ERP）。

与现有 SphereDiff 的区别：现有方法从噪声出发对整个球面同时去噪；新方法从一张已有透视图出发，以**已生成区域作为 control/context**，逐步向外延伸覆盖整个球面。

---

## 输入

| 参数 | 说明 |
|------|------|
| `input_image` | PIL 图像，单张透视图 |
| `input_fov` | 输入图的视场角 (FOV_h, FOV_w)，默认 (80, 80) 度 |
| `input_view_theta` | 输入图相机方位角（水平），度，默认 0 |
| `input_view_phi` | 输入图相机仰角（垂直），度，默认 0 |
| `prompt` | 文本描述，用于指导全景生成 |
| `negative_prompt` | 负向提示词，可选 |

## 输出

- ERP 格式全景图（默认 4096×2048）

---

## 核心流程

### 1. 初始化球面表示

1. 使用 `fibonacci_sphere(N)` 生成 N 个球面点（同现有代码）
2. 用 VAE 将输入透视图编码为 latent `x0`
3. 用 `paste_perspective_to_erp_rectangle` 逻辑的逆向操作将 `x0` 投影到球面点上，标记这些点为"已覆盖"
4. 维护 `sphere_latents`（每个球面点的 latent 值）和 `sphere_covered`（覆盖标记）

### 2. 贪心视角采样策略

目标：以最少步数覆盖整个球面，同时每步与已覆盖区域有足够重叠（保证一致性）。

```
候选视角集合 V = horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()
used = {initial_view}
covered = initial_covered_points

while coverage(covered) < 99%:
    best_view = None
    best_score = -inf

    for v in V \ used:
        v_points = get_visible_sphere_points(v, fov)
        overlap = |v_points ∩ covered| / |v_points|
        new_pts  = |v_points \ covered|

        if overlap >= min_overlap_ratio (default 0.25):
            score = new_pts  # 最大化新覆盖
            if score > best_score:
                best_view = v
                best_score = score

    if best_view is None:
        # 放宽条件：找最近的未覆盖区域
        best_view = argmax_v(|v_points \ covered|)

    used.add(best_view)
    执行该视角的生成（见步骤 3）
    covered.update(v_points)
```

**重叠参数调整**：
- `min_overlap_ratio = 0.25`：保证边界一致性
- FOV = 80°，overlap = 40%：每步有效新增约 50% FOV 面积
- 预计总步数：约 25-40 步覆盖整个球面

### 3. 单视角生成（RePaint 式蒙版去噪）

对每个新视角 `v`：

**a. 准备输入 latent**
```
visible_pts = get_visible_sphere_points(v)
known_pts   = visible_pts ∩ covered       # 已有内容
unknown_pts = visible_pts \ covered       # 待生成内容

# 提取已知区域的球面 latent
known_latents = sphere_latents[known_pts]  # (K, C)
```

**b. 构建初始 latent（纯噪声）**
```
x_t = randn(latent_h, latent_w)   # 全噪声初始化
```

**c. 去噪循环（Flux flow matching）**
对每个 timestep t:
```
# 1. 标准去噪步
x_pred = flux_transformer(x_t, t, prompt)
x_t    = scheduler.step(x_pred, t, x_t)

# 2. RePaint：将已知区域的 latent 重新注入
#    (在当前噪声水平下恢复已知区域)
t_normalized = t / 1000.0
x_known_noisy = (1 - t_normalized) * known_latents + t_normalized * noise
x_t[known_pts_indices] = x_known_noisy
```

**d. 更新球面**
```
# 只更新原来未覆盖的点（blending 方式）
sphere_latents[unknown_pts] = x_t[unknown_pts_indices]
sphere_covered.update(unknown_pts)
```

### 4. 最终解码

遍历所有视角，从球面 latent 提取透视图，VAE 解码，用 `paste_perspective_to_erp_rectangle` 混合（Gaussian 权重 blending）到 ERP 图上。

---

## 实现文件

| 文件 | 说明 |
|------|------|
| `pipelines_ours/pipeline_perspective_to_panorama_flux.py` | 主 pipeline 类 |
| `generate_panorama_from_perspective.py` | 命令行入口 |

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_spherical_points` | 26500 | 球面采样点数 |
| `min_overlap_ratio` | 0.25 | 最小重叠比例 |
| `num_inference_steps` | 28 | 每个视角去噪步数 |
| `guidance_scale` | 3.5 | 文本引导强度 |
| `weighted_average_temperature` | 0.1 | blending 温度参数 |
| `erp_height` / `erp_width` | 2048 / 4096 | 输出分辨率 |
| `repaint_strength` | 0.7 | 已知区域保留强度（越高=越保留） |

---

## 与现有代码的复用

- `SphericalFunctions.fibonacci_sphere()` ✓
- `SphericalFunctions.horizontal_and_vertical_view_dirs_v3_fov_xy_dense_equator()` ✓
- `SphericalFunctions.dynamic_laetent_sampling()` ✓
- `SphericalFunctions.paste_perspective_to_erp_rectangle()` ✓（最终混合）
- `SphericalFluxPipeline._pack_latents_for_spherical()` ✓
- `SphericalFluxPipeline._unpack_latents_for_spherical()` ✓
