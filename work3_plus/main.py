import taichi as ti
import numpy as np

# 1. 初始化 GPU
ti.init(arch=ti.gpu)

# 2. 常量与缓冲区预分配
WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 50
NUM_SEGMENTS = 1000

# 图像像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
# 曲线坐标缓冲区 (GPU)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
# 交互控制点缓冲 (GPU)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# 3. 纯 NumPy 递归计算 (完全避开 Taichi 运行时冲突)
def de_casteljau_np(points, t):
    arr = points.copy()
    n = len(arr)
    for i in range(1, n):
        for j in range(n - i):
            arr[j] = (1.0 - t) * arr[j] + t * arr[j + 1]
    return arr[0]

@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    # 抗锯齿：高斯权重衰减
    for k in range(n):
        pt = curve_points_field[k] * ti.Vector([WIDTH, HEIGHT])
        # 遍历 3x3 邻域像素
        for i in range(int(pt.x) - 1, int(pt.x) + 2):
            for j in range(int(pt.y) - 1, int(pt.y) + 2):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    dist_sq = (i - pt.x)**2 + (j - pt.y)**2
                    weight = ti.exp(-dist_sq * 0.5)
                    pixels[i, j] += ti.Vector([0.2, 0.7, 1.0]) * weight

def main():
    window = ti.ui.Window("Bezier Curve (Stable Version)", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points_list = [] # 存储为普通的列表
    
    while window.running:
        # 事件处理
        for e in window.get_events(ti.ui.PRESS):
            if e.key == 'c': 
                control_points_list = []
            elif e.key == ti.ui.LMB:
                pos = window.get_cursor_pos()
                if pos[0] is not None and len(control_points_list) < MAX_CONTROL_POINTS:
                    control_points_list.append(np.array(pos, dtype=np.float32))

        # 渲染循环
        clear_pixels()
        
        current_count = len(control_points_list)
        if current_count >= 2:
            pts_array = np.array(control_points_list)
            # CPU 采样
            sampled = [de_casteljau_np(pts_array, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS + 1)]
            curve_points_field.from_numpy(np.array(sampled, dtype=np.float32))
            draw_curve_kernel(NUM_SEGMENTS + 1)
        
        canvas.set_image(pixels)
        
        # 绘制控制点 (将列表填入对象池)
        if current_count > 0:
            gui_data = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            gui_data[:current_count] = np.array(control_points_list)
            gui_points.from_numpy(gui_data)
            canvas.circles(gui_points, radius=0.006, color=(0.8, 0.2, 0.2))
        
        window.show()

if __name__ == '__main__':
    main()