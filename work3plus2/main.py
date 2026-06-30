import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
MAX_CONTROL_POINTS = 50
NUM_SEGMENTS = 1000

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# 1. B-Spline 基矩阵计算函数
def get_bspline_point(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    # 三次均匀 B-Spline 基矩阵 (Basis Matrix)
    m = np.array([
        [-1,  3, -3, 1],
        [ 3, -6,  3, 0],
        [-3,  0,  3, 0],
        [ 1,  4,  1, 0]
    ]) / 6.0
    T = np.array([t3, t2, t, 1.0])
    coeffs = T @ m
    return coeffs[0]*p0 + coeffs[1]*p1 + coeffs[2]*p2 + coeffs[3]*p3

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
def draw_curve_kernel(n: ti.i32, color_mode: ti.i32):
    for k in range(n):
        pt = curve_points_field[k] * ti.Vector([WIDTH, HEIGHT])
        for i in range(int(pt.x) - 1, int(pt.x) + 2):
            for j in range(int(pt.y) - 1, int(pt.y) + 2):
                if 0 <= i < WIDTH and 0 <= j < HEIGHT:
                    dist_sq = (i - pt.x)**2 + (j - pt.y)**2
                    weight = ti.exp(-dist_sq * 0.5)
                    # 贝塞尔用蓝色，B-Spline 用紫色区分
                    color = ti.Vector([0.2, 0.7, 1.0]) if color_mode == 0 else ti.Vector([0.8, 0.2, 0.8])
                    pixels[i, j] += color * weight

def main():
    window = ti.ui.Window("Bezier vs B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    control_points_list = []
    is_bspline = False
    
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == 'c': control_points_list = []
            elif e.key == 'b': is_bspline = not is_bspline
            elif e.key == ti.ui.LMB:
                pos = window.get_cursor_pos()
                if pos[0] is not None and len(control_points_list) < MAX_CONTROL_POINTS:
                    control_points_list.append(np.array(pos, dtype=np.float32))

        clear_pixels()
        current_count = len(control_points_list)
        
        if current_count >= 2:
            sampled = []
            if is_bspline and current_count >= 4:
                # B-Spline 采样：分段拼接
                for i in range(current_count - 3):
                    for j in range(NUM_SEGMENTS // (current_count - 3) + 1):
                        sampled.append(get_bspline_point(
                            control_points_list[i], control_points_list[i+1],
                            control_points_list[i+2], control_points_list[i+3], j/100
                        ))
            else:
                # 贝塞尔采样
                pts_array = np.array(control_points_list)
                sampled = [de_casteljau_np(pts_array, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS + 1)]
            
            curve_points_field.from_numpy(np.array(sampled, dtype=np.float32))
            draw_curve_kernel(len(sampled), int(is_bspline))
        
        canvas.set_image(pixels)
        # ... (后续绘制控制点的代码保持不变)
        window.show()

if __name__ == '__main__':
    main()