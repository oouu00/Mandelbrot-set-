import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import (
    Tk,
    Label,
    Entry,
    Button,
    StringVar,
    filedialog,
    messagebox,
    OptionMenu,
    Frame,
)
import time
import traceback
import gmpy2
from gmpy2 import mpfr, mpc, get_context, log10
import pyopencl as cl
from pyopencl import array
from multiprocessing import Process, Queue, cpu_count
import tempfile

import subprocess  # 确保导入 subprocess 模块

from threading import Thread
from tkinter import BooleanVar
from tkinter import Checkbutton


# 并行计算配置
WORKERS = max(cpu_count() - 1, 1)  # 使用CPU核心-1

# 配置Matplotlib
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 全局控制变量
generating = False
current_task = 0
total_tasks = 0


# 获取可用的CPU核心数
MAX_WORKERS = WORKERS


# ================== 核心计算模块 =======


def initialize_opencl():
    """初始化 OpenCL 环境并编译内核"""
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("没有找到可用的 OpenCL 平台")

    platform = platforms[0]
    print(f"选择的 OpenCL 平台: {platform.name}")

    devices = platform.get_devices()
    if not devices:
        raise RuntimeError("没有找到可用的 OpenCL 设备")

    device = devices[0]
    print(f"选择的 OpenCL 设备: {device.name}")

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(
        ctx, properties=cl.command_queue_properties.PROFILING_ENABLE
    )

    # 编译内核代码
    kernel_code = """
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void mandelbrot(
        __global float *output,
        const double x_min, const double x_max,
        const double y_min, const double y_max,
        const int width, const int height,
        const int max_iter,const double c_x, const double c_y, const int Julia)
        
        
        
    {
        int j = get_global_id(0);
        int i = get_global_id(1);
        
        if (i >= height || j >= width) return;
        
        double x = x_min + (double)j / (width - 1) * (x_max - x_min);
        double y = y_min + (double)i / (height - 1) * (y_max - y_min);
               
        
        // 根据 Julia 参数决定 c_x 和 c_y 的值
        double cr = (Julia == 0) ? c_x : x;
        double ci = (Julia == 0) ? c_y : y;
    
        double zr = x;
        double zi = y;
        int n = 0;
        double zr2, zi2;
        
        while (n < max_iter) {
            zr2 = zr * zr;
            zi2 = zi * zi;
            if (zr2 + zi2 > 4.0) break;
            
            zi = 2 * zr * zi + ci;
            zr = zr2 - zi2 + cr;
            n++;
        }
        
        float result = (n < max_iter) ? n + 1 - log(log(sqrt(zr2 + zi2)) / log(2.0)) / log(2.0) : max_iter;
        output[i * width + j] = result;
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    print("内核编译完成")

    return ctx, queue, prg


# ================== GUI界面模块 ==================
class MandelbrotGenerator:

    def gmp_high_precision_compute(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
        width,
        height,
        max_iter,
        precision,
        current_task,
    ):
        """高精度计算 Mandelbrot 集"""
        # 从 MandelbrotGenerator 类的实例中获取当前参数
        current_params = self.validate_parameters()  # 获取当前参数
        if current_params is None:
            raise ValueError("参数验证失败")

        # 计算二进制精度
        binary_precision = int(precision * 3.324) + 16
        print(f"二进制精度: {binary_precision}")

        thread_count = WORKERS
        print(f"线程数: {thread_count}")
        print(f"初始缩放比例: {current_params['initial_scale']}")

        # 计算当前缩放比例
        current_scale2 = mpfr(current_params["initial_scale"]) / (
            mpfr(current_params["zoom_factor"]) ** (current_task - 1)
        )

        percentile3 = float(self.percentile_entry.get())  # 从输入框获取百分位数

        percentile2 = log10(percentile3) / 2  # 使用 gmpy2 的 log10 函数
        if self.julia_var.get():
            julia = 1
        else:
            julia = 0

        # 将参数写入临时文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "mandelbrot.txt")

        with open(file_path, "w") as temp_file:
            temp_file.write(f"{thread_count}\n")
            temp_file.write(f"{binary_precision}\n")
            temp_file.write(f"{current_params['center_x']}\n")
            temp_file.write(f"{current_params['center_y']}\n")
            temp_file.write(f"{width}\n")
            temp_file.write(f"{height}\n")
            temp_file.write(f"{str(current_scale2)}\n")
            temp_file.write(f"{current_params['focus_x']}\n")
            temp_file.write(f"{current_params['focus_y']}\n")
            temp_file.write(f"{current_params['max_iter']}\n")
            temp_file.write(f"{percentile2}\n")
            temp_file.write(f"{current_params['c_x']}\n")
            temp_file.write(f"{current_params['c_y']}\n")
            temp_file.write(f"{julia}\n")

            temp_file.flush()  # 确保内容写入磁盘

        # 切换工作目录到文件所在路径
        os.chdir(script_dir)  # 切换到脚本目录

        # 调用 Mandelbrot.exe 并传递临时文件路径
        command = f"Mandelbrot.exe mandelbrot.txt"
        print(f"执行命令: {command}")

        self.mandelbrot_process = subprocess.Popen(command, shell=True)  # 保存进程对象
        while self.mandelbrot_process.poll() is None:
            if not generating:
                # 强行终止 Mandelbrot.exe 进程
                subprocess.run("taskkill /F /IM Mandelbrot.exe", shell=True)
                raise Exception("计算已终止")
            time.sleep(0.2)
        # 处理结果...

        print("命令完成")

        # 加载结果文件
        output = self.load_iteration_array_from_file(width, height)
        if output is None:
            raise FileNotFoundError("未找到 iteration_array.bin 文件或文件格式不正确")
        return output

    def load_iteration_array_from_file(self, width, height):
        """从文件中加载迭代次数数组"""
        file_path = "iteration_array.bin"
        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在")
            return None

        try:
            # 打开文件并读取数据
            with open(file_path, "rb") as file:
                # 假设文件存储的是整数的二维数组
                data = np.fromfile(file, dtype=np.int32)
                if len(data) != width * height:
                    print(f"文件 {file_path} 的数据大小与预期不符")
                    return None
                # 将数据重塑为二维数组
                output = data.reshape((height, width))

                return output
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
            return None

    def create_buttons(self, parent):
        """创建区域切换按钮"""
        buttons_frame = Frame(parent, padx=10, pady=5)
        buttons_frame.grid(row=3, column=0, sticky="ew", padx=5)

        regions = [
            (
                "海马谷",
                -0.74515135254160040270968367987368282086820830812505531306383690724472274612043,
                0.1301973420730631518534562967071010595298732429535045571757108286281891077748,
            ),
            ("花", -0.3906731935564842, 0.6243239460184142),
            ("螺丝线", -1.999944284562545, 7.674460749033766e-16),
            ("羽毛", -1.49005323192932, 0.00119631499924505),
            ("螺旋", -0.6888873943437659, 0.2810053792512903),
            ("螺旋2", 0.2513567236163006, -9.220290965617807e-05),
        ]

        for idx, (name, x, y) in enumerate(regions):
            btn = Button(
                buttons_frame,
                text=name,
                width=3,
                command=lambda x=x, y=y: self.set_focus(x, y),
            )
            btn.grid(row=idx // 2, column=idx % 2, padx=3, pady=1.5)

    # 新增设置焦点坐标的方法（直接传递x和y数值）
    def set_focus(self, x_val, y_val):
        """直接设置焦点坐标并更新图像"""
        try:
            # 强制保留15位小数（根据需求可调整）
            self.center_x_entry.delete(0, "end")
            self.center_x_entry.insert(0, f"{x_val}")

            self.center_y_entry.delete(0, "end")
            self.center_y_entry.insert(0, f"{y_val}")

        except Exception as e:
            messagebox.showerror("输入错误", f"坐标设置失败: {str(e)}")

    def safe_compute_mandelbrot(
        self,
        x_min,
        x_max,
        y_min,
        y_max,
        width,
        height,
        max_iter,
        precision_threshold,
        current_task,
        c_x,
        c_y,
        precision=50,
    ):
        # """智能计算分发函数"""
        try:
            if precision > precision_threshold:
                # 高精度并行模式（原有实现）
                self.current_method = "gmp高精度计算"  # 高精度模式
                # 从 MandelbrotGenerator 类的实例中获取当前参数
                return self.gmp_high_precision_compute(
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    width,
                    height,
                    max_iter,
                    precision,
                    current_task,
                )

            else:
                try:
                    # 尝试使用 GPU 加速
                    self.current_method = "OpenCL"  # 尝试OpenCL
                    return self.opencl_mandelbrot(
                        x_min, x_max, y_min, y_max, width, height, max_iter, c_x, c_y
                    )
                except:
                    # 使用普通 CPU 计算
                    self.current_method = "opencl失败，切gmp高精度计算"
                    return self.gmp_high_precision_compute(
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        width,
                        height,
                        max_iter,
                        precision,
                        current_task,
                    )

        except Exception as e:
            print(f"计算错误: {str(e)}")
            raise

    def opencl_mandelbrot(
        self, x_min, x_max, y_min, y_max, width, height, max_iter, c_x, c_y
    ):
        """使用 OpenCL 计算 Mandelbrot 集 (同步优化版)"""

        if self.julia_var.get():
            julia = 0
        else:
            julia = 1

        print("开始opencl计算")
        # 创建输出缓冲区和映射
        output = np.empty((height, width), dtype=np.float32)
        output_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR,
            size=output.nbytes,
        )

        # 预计算：缩小 10 倍的宽度和高度
        pre_width = max(width // 10, 1)  # 防止宽度或高度为 0
        pre_height = max(height // 10, 1)  # 防止宽度或高度为 0
        pre_output = np.empty((pre_height, pre_width), dtype=np.float32)
        pre_output_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR,
            size=pre_output.nbytes,
        )
        print("准备发内核")
        # 同步执行预计算内核
        global_size = (pre_width, pre_height)  # 使用2D全局工作尺寸
        pre_kernel_event = self.prg.mandelbrot(
            self.queue,
            global_size,
            None,  # 自动选择工作组大小
            pre_output_buf,
            np.float64(x_min),
            np.float64(x_max),
            np.float64(y_min),
            np.float64(y_max),
            np.int32(pre_width),
            np.int32(pre_height),
            np.int32(max_iter),
            np.float64(c_x),
            np.float64(c_y),
            np.int32(julia),
        )

        print("数据已发送内核")
        pre_kernel_event.wait()  # 等待内核完成
        # 映射预计算结果
        mapped_pre_data, _ = cl.enqueue_map_buffer(
            self.queue,
            pre_output_buf,
            cl.map_flags.READ,
            0,
            pre_output.shape,
            pre_output.dtype,
        )
        np.copyto(pre_output, mapped_pre_data)

        # 去除所有重复的最大值
        max_iter_value = np.max(pre_output)  # 获取最大迭代次数的值
        pre_output[pre_output == max_iter_value] = 0  # 将最大值替换为 NaN

        percentile = float(self.percentile_entry.get())  # 从输入框获取百分位数

        percentile_log = log10(percentile) * 50  # 使用 gmpy2 的 log10 函数并校准
        percentile_log = float(percentile_log)  # 从输入框获取百分位数

        # 计算新的最大迭代次数（取 98% 的位置）
        new_max_iter = int(np.percentile(pre_output, percentile_log))

        new_max_iter = max(new_max_iter, 100)  # 确保 new_max_iter 不小于 100
        print(f"预计算完成，新的最大迭代次数: {new_max_iter}")

        # 正式计算：使用新的最大迭代次数
        global_size = (width, height)  # 使用2D全局工作尺寸
        kernel_event = self.prg.mandelbrot(
            self.queue,
            global_size,
            None,  # 自动选择工作组大小
            output_buf,
            np.float64(x_min),
            np.float64(x_max),
            np.float64(y_min),
            np.float64(y_max),
            np.int32(width),
            np.int32(height),
            np.int32(new_max_iter),
            np.float64(c_x),
            np.float64(c_y),
            np.int32(julia),
        )
        kernel_event.wait()  # 等待内核完成

        # 映射正式计算结果
        mapped_data, _ = cl.enqueue_map_buffer(
            self.queue, output_buf, cl.map_flags.READ, 0, output.shape, output.dtype
        )
        np.copyto(output, mapped_data)

        return output

    def get_zoom_factor(self):
        """从输入框中获取最新的放大系数"""
        try:
            zoom_factor = float(self.zoom_factor_entry.get())
            if zoom_factor <= 0:
                raise ValueError("放大系数必须大于0")
            return zoom_factor
        except ValueError as e:
            messagebox.showerror("输入错误", f"放大系数无效: {str(e)}")
            raise

    def on_canvas_release(self, event):
        """处理画布上的鼠标松开事件"""
        if not generating and event.inaxes:
            x_pixel, y_pixel = event.xdata, event.ydata  # 获取鼠标松开时的坐标
            # 获取当前显示范围
            # x_min, x_max = self.current_metadata['x_range']
            # y_min, y_max = self.current_metadata['y_range']

            # 将像素坐标转换为复平面上的实际坐标
            # center_x = x_pixel
            # center_y = y_pixel

            default_width = int(self.width_entry.get())  # 从输入框获取宽度
            default_height = int(self.height_entry.get())  # 从输入框获取高度

            # 更新焦点位置
            focus_x = x_pixel / default_width
            focus_y = y_pixel / default_height

            if self.select_c_var.get():  # 如果勾选了“画布选择c坐标”
                pass  # 如果没有具体逻辑，可以用 pass 占位
            else:

                # 更新焦点位置输入框
                self.focus_x_entry.delete(0, "end")
                self.focus_x_entry.insert(0, f"{focus_x}")

                self.focus_y_entry.delete(0, "end")
                self.focus_y_entry.insert(0, f"{focus_y}")

                # 可选：自动重新生成 Mandelbrot 集
                self.start_generation()

    def on_canvas_click(self, event):
        """处理画布上的鼠标点击事件"""
        if not generating and event.inaxes:
            # 获取点击位置的数据坐标
            x_pixel, y_pixel = event.xdata, event.ydata

            # 获取当前显示范围
            x_min, x_max = self.current_metadata["x_range"]
            y_min, y_max = self.current_metadata["y_range"]
            # 将像素坐标转换为复平面上的实际坐标
            center_x = x_pixel
            center_y = y_pixel

            current_scalex = mpfr(x_max) - mpfr(x_min)  # 当前显示范围的宽度
            current_scaley = mpfr(y_max) - mpfr(y_min)  # 当前显示范围的高度

            default_width = int(self.width_entry.get())  # 从输入框获取宽度
            default_height = int(self.height_entry.get())  # 从输入框获取高度

            # 更新焦点位置
            cc_x = mpfr(x_pixel) / mpfr(default_width) * 4 - 2
            cc_y = mpfr(y_pixel) / mpfr(default_height) * 4 - 2

            # 计算点击位置的实际坐标
            center_x = mpfr(x_pixel) * mpfr(current_scalex) / mpfr(
                default_width
            ) + mpfr(x_min)
            center_y = mpfr(y_pixel) * mpfr(current_scaley) / mpfr(
                default_height
            ) + mpfr(y_min)

            # 检查是否启用了“画布选择c坐标”选项
            if self.select_c_var.get():  # 如果勾选了“画布选择c坐标”
                # 更新 c_x 和 c_y 的值
                self.c_x_entry.delete(0, "end")
                self.c_x_entry.insert(0, str(cc_x))

                self.c_y_entry.delete(0, "end")
                self.c_y_entry.insert(0, str(cc_y))
                self.start_generation()

            else:
                # 否则更新焦点数值坐标

                # 更新宽度数值

                current_scale = current_scalex / mpfr(
                    self.get_zoom_factor()
                )  # 再除以一次放大系数

                self.scale_entry.delete(0, "end")
                self.scale_entry.insert(0, str(current_scale))
                self.center_x_entry.delete(0, "end")
                self.center_x_entry.insert(0, str(center_x))

                self.center_y_entry.delete(0, "end")
                self.center_y_entry.insert(0, str(center_y))
                # self.start_generation()

            # 可选：自动重新生成 Mandelbrot 集
            # self.start_generation()

    def __init__(self, root):
        self.root = root
        self.current_image = None  # 初始化为 None 或其他默认值
        self.current_metadata = {}  # 初始化为一个空字典
        self.mandelbrot_process = None  # 用于存储 Mandelbrot.exe 的进程对象

        self.setup_ui()
        self.setup_plot()
        # 初始化 OpenCL 环境
        self.ctx, self.queue, self.prg = initialize_opencl()
        self.current_method = "OpenCL"  # 默认使用Numba
        self.last_click_time = 0  # 上一次点击的时间
        self.is_double_click = False  # 是否为双击

    def setup_ui(self):
        """初始化用户界面"""
        self.root.title("Mandelbrot集图片生成器")
        self.root.geometry("1200x800")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # 控制面板（改为Frame容器）
        control_frame = Frame(self.root, padx=10, pady=10)
        control_frame.grid(row=0, column=0, sticky="nsew")
        control_frame.grid_columnconfigure(1, weight=1)  # 新增列权重配置

        # 参数输入控件
        self.create_inputs(control_frame)

        # 状态栏
        self.status_var = StringVar()
        self.status_var.set("就绪")
        status_bar = Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief="sunken",
            anchor="w",
            font=("微软雅黑", 9),
            bg="#f0f0f0",
        )
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def create_inputs(self, parent):
        """创建输入控件"""
        row = 0
        Label(parent, text="分辨率", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.width_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.width_entry.insert(0, "800")
        self.width_entry.grid(row=row, column=1, padx=2)
        Label(parent, text="×", font=("微软雅黑", 10)).grid(row=row, column=2)
        self.height_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.height_entry.insert(0, "600")
        self.height_entry.grid(row=row, column=3, padx=2)

        row += 1
        Label(parent, text="焦点数值坐标 X", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.center_x_entry = Entry(parent, width=20, font=("微软雅黑", 10))
        self.center_x_entry.insert(
            0,
            "-0.74515135254160040270968367987368282086820830812505531306383690724472274612043",
        )
        self.center_x_entry.grid(row=row, column=1, columnspan=3, padx=2)

        row += 1
        Label(parent, text="焦点数值坐标 Y", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.center_y_entry = Entry(parent, width=20, font=("微软雅黑", 10))
        self.center_y_entry.insert(
            0,
            "0.1301973420730631518534562967071010595298732429535045571757108286281891077748",
        )
        self.center_y_entry.grid(row=row, column=1, columnspan=3, padx=2)

        # 添加朱利亚开关
        row += 2
        self.julia_var = BooleanVar()
        self.julia_checkbox = Checkbutton(
            parent,
            text="启用朱利亚集",
            variable=self.julia_var,
            command=self.toggle_julia_mode,
            font=("微软雅黑", 10),
        )
        self.julia_checkbox.grid(row=row, column=0, columnspan=5, sticky="w", pady=2)
        row += 1
        # 添加“画布上选择c坐标”开关
        self.select_c_var = BooleanVar()  # 创建一个布尔变量
        self.select_c_checkbox = Checkbutton(
            parent,
            text="画布上选择c坐标",
            variable=self.select_c_var,
            font=("微软雅黑", 10),
        )
        self.select_c_checkbox.grid(row=row, column=0, columnspan=5, sticky="w", pady=2)

        row += 1
        self.copy_button = Button(
            parent,
            text="复制焦点坐标到 c 坐标",
            command=self.copy_center_to_c,
            font=("微软雅黑", 10),
        )
        self.copy_button.grid(row=row, column=0, columnspan=5, pady=5, sticky="we")

        row += 1

        Label(parent, text="c坐标 X", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.c_x_entry = Entry(parent, width=20, font=("微软雅黑", 10))
        self.c_x_entry.insert(
            0,
            "-0.74515135254160040270968367987368282086820830812505531306383690724472274612043",
        )
        self.c_x_entry.grid(row=row, column=1, columnspan=3, padx=2)
        self.c_x_entry.config(state="disabled")  # 默认禁用

        row += 1
        Label(parent, text="c坐标 Y", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.c_y_entry = Entry(parent, width=20, font=("微软雅黑", 10))
        self.c_y_entry.insert(
            0,
            "0.1301973420730631518534562967071010595298732429535045571757108286281891077748",
        )
        self.c_y_entry.grid(row=row, column=1, columnspan=3, padx=2)
        self.c_y_entry.config(state="disabled")  # 默认禁用

        row += 1
        Label(parent, text="宽度数值", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.scale_entry = Entry(parent, width=20, font=("微软雅黑", 10))
        self.scale_entry.insert(0, "3.0000000")
        self.scale_entry.grid(row=row, column=1, columnspan=3, padx=2)

        row += 1
        Label(parent, text="动态迭代数上限", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.max_iter_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.max_iter_entry.insert(0, "9999")
        self.max_iter_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="动态迭代阈值(0-100)", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.percentile_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.percentile_entry.insert(0, "90")  # 默认值为 90
        self.percentile_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="生成数量", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.num_images_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.num_images_entry.insert(0, "4")
        self.num_images_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="放大系数", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.zoom_factor_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.zoom_factor_entry.insert(0, "1.5")
        self.zoom_factor_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="焦点位置坐标 X", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.focus_x_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.focus_x_entry.insert(0, "0.5")  # 默认值为 0.5
        self.focus_x_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="焦点位置坐标 Y", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.focus_y_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.focus_y_entry.insert(0, "0.5")  # 默认值为 0.5
        self.focus_y_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="切换cpu阈值", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.precision_threshold_entry = Entry(parent, width=8, font=("微软雅黑", 10))
        self.precision_threshold_entry.insert(0, "11")  # 默认值为 11
        self.precision_threshold_entry.grid(row=row, column=1, padx=2)

        row += 1
        Label(parent, text="颜色映射", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.colormap_var = StringVar()
        self.colormap_var.set("jet")  # 默认值
        colormaps = [
            "Accent",
            "Accent_r",
            "Blues",
            "Blues_r",
            "BrBG",
            "BrBG_r",
            "BuGn",
            "BuGn_r",
            "BuPu",
            "BuPu_r",
            "CMRmap",
            "CMRmap_r",
            "Dark2",
            "Dark2_r",
            "GnBu",
            "GnBu_r",
            "Grays",
            "Greens",
            "Greens_r",
            "Greys",
            "Greys_r",
            "OrRd",
            "OrRd_r",
            "Oranges",
            "Oranges_r",
            "PRGn",
            "PRGn_r",
            "Paired",
            "Paired_r",
            "Pastel1",
            "Pastel1_r",
            "Pastel2",
            "Pastel2_r",
            "PiYG",
            "PiYG_r",
            "PuBu",
            "PuBuGn",
            "PuBuGn_r",
            "PuBu_r",
            "PuOr",
            "PuOr_r",
            "PuRd",
            "PuRd_r",
            "Purples",
            "Purples_r",
            "RdBu",
            "RdBu_r",
            "RdGy",
            "RdGy_r",
            "RdPu",
            "RdPu_r",
            "RdYlBu",
            "RdYlBu_r",
            "RdYlGn",
            "RdYlGn_r",
            "Reds",
            "Reds_r",
            "Set1",
            "Set1_r",
            "Set2",
            "Set2_r",
            "Set3",
            "Set3_r",
            "Spectral",
            "Spectral_r",
            "Wistia",
            "Wistia_r",
            "YlGn",
            "YlGnBu",
            "YlGnBu_r",
            "YlGn_r",
            "YlOrBr",
            "YlOrBr_r",
            "YlOrRd",
            "YlOrRd_r",
            "afmhot",
            "afmhot_r",
            "autumn",
            "autumn_r",
            "binary",
            "binary_r",
            "bone",
            "bone_r",
            "brg",
            "brg_r",
            "bwr",
            "bwr_r",
            "cividis",
            "cividis_r",
            "cool",
            "cool_r",
            "coolwarm",
            "coolwarm_r",
            "copper",
            "copper_r",
            "cubehelix",
            "cubehelix_r",
            "flag",
            "flag_r",
            "gist_earth",
            "gist_earth_r",
            "gist_gray",
            "gist_gray_r",
            "gist_grey",
            "gist_heat",
            "gist_heat_r",
            "gist_ncar",
            "gist_ncar_r",
            "gist_rainbow",
            "gist_rainbow_r",
            "gist_stern",
            "gist_stern_r",
            "gist_yarg",
            "gist_yarg_r",
            "gist_yerg",
            "gnuplot",
            "gnuplot2",
            "gnuplot2_r",
            "gnuplot_r",
            "gray",
            "gray_r",
            "grey",
            "hot",
            "hot_r",
            "hsv",
            "hsv_r",
            "inferno",
            "inferno_r",
            "jet",
            "jet_r",
            "magma",
            "magma_r",
            "nipy_spectral",
            "nipy_spectral_r",
            "ocean",
            "ocean_r",
            "pink",
            "pink_r",
            "plasma",
            "plasma_r",
            "prism",
            "prism_r",
            "rainbow",
            "rainbow_r",
            "seismic",
            "seismic_r",
            "spring",
            "spring_r",
            "summer",
            "summer_r",
            "tab10",
            "tab10_r",
            "tab20",
            "tab20_r",
            "tab20b",
            "tab20b_r",
            "tab20c",
            "tab20c_r",
            "terrain",
            "terrain_r",
            "turbo",
            "turbo_r",
            "twilight",
            "twilight_r",
            "twilight_shifted",
            "twilight_shifted_r",
            "viridis",
            "viridis_r",
            "winter",
            "winter_r",
        ]
        self.colormap_menu = OptionMenu(parent, self.colormap_var, *colormaps)
        self.colormap_menu.grid(
            row=row, column=1, padx=2, sticky="ew"
        )  # 添加sticky参数

        row += 1
        Label(parent, text="保存路径", font=("微软雅黑", 10)).grid(
            row=row, column=0, sticky="w", pady=2
        )
        self.save_path_var = StringVar()
        Entry(
            parent,
            textvariable=self.save_path_var,
            state="readonly",
            width=20,
            font=("微软雅黑", 10),
        ).grid(row=row, column=1, columnspan=3, padx=2)
        self.browse_button = Button(
            parent, text="浏览...", command=self.select_save_path, font=("微软雅黑", 9)
        )
        self.browse_button.grid(row=row, column=4, padx=2)

        row += 1
        self.generate_button = Button(
            parent,
            text="开始生成",
            command=self.toggle_generation,
            font=("微软雅黑", 10, "bold"),
            bg="#4CAF50",
            fg="white",
        )
        self.generate_button.grid(row=row, column=0, columnspan=5, pady=10, sticky="we")

        # 创建区域切换按钮
        self.create_buttons(parent)

    def copy_center_to_c(self):
        """将焦点数值坐标复制到 c 坐标输入框"""
        try:
            # 获取焦点数值坐标
            center_x = self.center_x_entry.get()
            center_y = self.center_y_entry.get()

            # 将焦点数值坐标设置到 c 坐标输入框
            self.c_x_entry.config(state="normal")
            self.c_x_entry.delete(0, "end")
            self.c_x_entry.insert(0, center_x)

            self.c_y_entry.config(state="normal")
            self.c_y_entry.delete(0, "end")
            self.c_y_entry.insert(0, center_y)

            # 如果朱利亚模式未启用，自动启用
            if not self.julia_var.get():
                self.julia_var.set(True)
                self.toggle_julia_mode()

            messagebox.showinfo("复制成功", "焦点坐标已复制到 c 坐标")
        except Exception as e:
            messagebox.showerror("复制失败", f"复制坐标时出错: {str(e)}")

    def toggle_julia_mode(self):
        """切换朱利亚模式"""
        if self.julia_var.get():
            # 启用c坐标输入框
            self.c_x_entry.config(state="normal")
            self.c_y_entry.config(state="normal")
        else:
            # 禁用c坐标输入框
            self.c_x_entry.config(state="disabled")
            self.c_y_entry.config(state="disabled")

    def setup_plot(self):
        """初始化绘图区域"""
        self.fig = plt.figure(figsize=(8, 6), dpi=200)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(
            row=0, column=1, sticky="nsew", padx=10, pady=10
        )
        self.canvas.mpl_connect("resize_event", self.on_resize)  # 添加窗口大小调整事件
        self.canvas.mpl_connect(
            "button_press_event", self.on_canvas_click
        )  # 添加鼠标点击事件
        self.canvas.mpl_connect(
            "button_release_event", self.on_canvas_release
        )  # 鼠标松开事件

    def toggle_inputs(self, state):
        """切换输入控件状态"""
        entries = [
            self.width_entry,
            self.height_entry,
            self.center_x_entry,
            self.center_y_entry,
            self.c_x_entry,
            self.c_y_entry,
            self.scale_entry,
            self.max_iter_entry,
            self.num_images_entry,
            self.zoom_factor_entry,
            self.focus_x_entry,
            self.focus_y_entry,
            self.percentile_entry,
            self.precision_threshold_entry,
        ]
        for entry in entries:
            entry.config(state="normal" if state else "disabled")
        self.browse_button.config(state="normal" if state else "disabled")

    def select_save_path(self):
        """选择保存路径"""
        path = filedialog.askdirectory()
        if path:
            self.save_path_var.set(path)

    def update_progress(self):
        if not generating:
            return
        self.root.after(
            0, lambda: self.status_var.set(f"进度: {current_task}/{total_tasks}")
        )

        if current_task == total_tasks:
            self.status_var.set("生成完成")
        self.root.update_idletasks()

    def update_canvas(self, output, metadata):
        """更新画布显示"""
        if output is None or "x_range" not in metadata or "y_range" not in metadata:
            print("跳过更新画布：缺少必要的数据或元数据。")
            return
        try:
            self.fig.clear()
            print("清除画布成功")
            selected_colormap = self.colormap_var.get()  # 获取用户选择的颜色映射

            ax = self.fig.add_subplot(111)
            # 使用默认值
            default_width = int(self.width_entry.get())  # 从输入框获取宽度
            default_height = int(self.height_entry.get())  # 从输入框获取高度

            ax.imshow(
                output,
                cmap=selected_colormap,
                extent=[0, default_width, 0, default_height],  # 使用默认的宽度和高度
                origin="lower",
            )

            # 计算迭代次数范围
            outmin_iter = int(np.min(output))
            outmax_iter = int(np.max(output))

            title = f"当前计算方法: {self.current_method}"
            title += f"点击放大,拖动移动\n"
            title += f"实际迭代范围: [{outmin_iter}, {outmax_iter}]\n"
            title += f"计算时间: {metadata['compute_time']:.2f}s | 当前精度: {metadata['precision']}位 (第{metadata['task_id']}帧)"

            ax.set_title(title)
            # ax.set_xlabel("实部")
            # ax.set_ylabel("虚部")
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("界面更新错误", str(e))
        self.root.update()

    def on_resize(self, event):
        """窗口大小调整时的回调函数"""
        self.update_canvas(self.current_image, self.current_metadata)
        if self.current_image is not None and self.current_metadata:
            self.update_canvas(self.current_image, self.current_metadata)

    def toggle_generation(self):
        """启动/停止生成"""
        global generating
        if generating:
            generating = False
            self.generate_button.config(text="开始生成", state="normal")
            self.status_var.set("已终止")
            self.toggle_inputs(True)
            self.toggle_julia_mode()
            # 强行终止 Mandelbrot.exe 进程
            if self.mandelbrot_process and self.mandelbrot_process.poll() is None:
                subprocess.run("taskkill /F /IM Mandelbrot.exe", shell=True)

        else:
            self.start_generation()

    def start_generation(self):
        """开始生成任务"""
        global generating, current_task, total_tasks

        try:
            # 验证用户输入的参数
            params = self.validate_parameters()
            print("参数验证完成，参数如下：")
            for key, value in params.items():
                print(f"    {key}: {value}")

            # 初始化生成任务
            self.initialize_generation(params)
            print("生成任务初始化完成。")

            # 创建任务队列
            tasks = self.create_tasks(params)
            print(f"任务队列创建完成，共有 {len(tasks)} 个任务。")

            # 处理任务队列
            # 启动后台线程
            self.generation_thread = Thread(
                target=self.process_tasks, args=(tasks, params)
            )
            self.generation_thread.start()

            print("所有任务处理完成。")

        except Exception as e:
            # 捕获异常并显示错误信息
            error_message = f"发生错误:\n{str(e)}\n{traceback.format_exc()}"
            print(f"生成过程中发生错误：\n{error_message}")
            messagebox.showerror("错误", error_message)
            self.cleanup_after_error()

    def validate_parameters(self):
        """验证输入参数"""
        params = {
            "width": self.width_entry.get(),  # 获取用户输入的图像宽度
            "height": self.height_entry.get(),  # 获取用户输入的图像高度
            "center_x": self.center_x_entry.get(),  # 获取用户输入的中心点 X 坐标
            "center_y": self.center_y_entry.get(),  # 获取用户输入的中心点 Y 坐标
            "c_x": self.c_x_entry.get(),  # 获取用户输入的中心点 X 坐标
            "c_y": self.c_y_entry.get(),  # 获取用户输入的中心点 Y 坐标
            "initial_scale": self.scale_entry.get(),  # 获取用户输入的初始缩放比例
            "max_iter": self.max_iter_entry.get(),  # 获取用户输入的最大迭代次数
            "num_images": self.num_images_entry.get(),  # 获取用户输入的生成图像数量
            "zoom_factor": self.zoom_factor_entry.get(),  # 获取用户输入的放大系数
            "save_path": self.save_path_var.get(),  # 获取用户选择的保存路径
            "focus_x": self.focus_x_entry.get(),  # 获取用户输入的焦点位置 X 比例
            "focus_y": self.focus_y_entry.get(),  # 获取用户输入的焦点位置 Y 比例
        }

        if any(
            v <= 0
            for v in [
                int(params["width"]),
                int(params["height"]),
                float(params["initial_scale"]),
                int(params["max_iter"]),
            ]
        ):
            raise ValueError("参数必须大于0")
        # if params['zoom_factor'] <= 1.0:
        # raise ValueError("放大系数必须大于1.0")
        if not os.path.exists(params["save_path"]):
            os.makedirs(params["save_path"], exist_ok=True)
        self.params = params  # 将 params 存储为类的属性
        return params

    def initialize_generation(self, params):
        """初始化生成任务"""
        global generating, current_task, total_tasks
        generating = True
        current_task = 0
        total_tasks = int(params["num_images"])
        self.generate_button.config(text="终止", state="normal")
        self.status_var.set("初始化中...")
        self.toggle_inputs(False)
        self.root.update()

    def create_tasks(self, params):
        """创建任务队列"""
        tasks = []
        current_scale = mpfr(params["initial_scale"])  # 使用 mpfr 类型
        zoom_factor = mpfr(params["zoom_factor"])
        focus_x = mpfr(params["focus_x"])
        focus_y = mpfr(params["focus_y"])

        # 设置 mpfr 的精度（可以根据需要调整）
        get_context().precision = 262144  # 例如，设置为 128 位二进制精度

        for i in range(int(params["num_images"])):
            # 使用 mpfr 进行高精度计算
            mpfr_scale = mpfr(current_scale) * mpfr(0.9)
            mpfr_log10 = log10(mpfr_scale)
            precision = max(1, int(-mpfr_log10))

            # 打印当前精度计算结果
            print(int(-mpfr_log10))

            # 重新设置 mpfr 的精度，多10位
            if (precision * 3.3) > 0:
                get_context().precision = int(precision * 3.324) + 16

            # 创建任务字典，所有数值均使用 mpfr 类型
            task = {
                "task_id": i + 1,
                "width": int(params["width"]),  # 整数类型保持不变
                "height": int(params["height"]),  # 整数类型保持不变
                "x_min": mpfr(params["center_x"])
                - current_scale / 2
                + (mpfr(0.5) - focus_x) * current_scale,
                "x_max": mpfr(params["center_x"])
                + current_scale / 2
                + (mpfr(0.5) - focus_x) * current_scale,
                "y_min": mpfr(params["center_y"])
                - current_scale / (2 * mpfr(params["width"]) / mpfr(params["height"]))
                + (mpfr(0.5) - focus_y)
                * current_scale
                / (mpfr(params["width"]) / mpfr(params["height"])),
                "y_max": mpfr(params["center_y"])
                + current_scale / (2 * mpfr(params["width"]) / mpfr(params["height"]))
                + (mpfr(0.5) - focus_y)
                * current_scale
                / (mpfr(params["width"]) / mpfr(params["height"])),
                "max_iter": int(params["max_iter"]),  # 整数类型保持不变
                "c_x": mpfr(params["c_x"]),
                "c_y": mpfr(params["c_y"]),
                "precision": precision,
            }
            current_scale /= mpfr(self.get_zoom_factor())
            print("task创建成功")
            tasks.append(task)  # 确保任务被添加到任务列表中
        return tasks

    def process_tasks(self, tasks, params):
        """处理任务队列"""
        global generating, current_task
        for task in tasks:
            if not generating:
                break

            current_task += 1
            self.status_var.set(f"正在生成第{current_task}帧...")
            self.update_progress()

            # 执行计算
            print("计算开始时间")
            print(time.time())
            start_time = time.time()
            precision_threshold = int(self.precision_threshold_entry.get())
            output = self.safe_compute_mandelbrot(
                task["x_min"],
                task["x_max"],
                task["y_min"],
                task["y_max"],
                task["width"],
                task["height"],
                task["max_iter"],
                precision_threshold,
                current_task,
                task["c_x"],
                task["c_y"],
                task["precision"],
            )
            compute_time = time.time() - start_time
            print("计算完成时间")
            print(time.time())
            # 保存结果
            print("保存前时间")
            print(time.time())
            # 假设 params 和 task 已经定义，output 是要保存的图像数据
            filename_base = os.path.join(params["save_path"], "Mandelbrot")
            file_extension = ".png"
            filename = f"{filename_base}{file_extension}"
            counter = 1

            # 检查文件是否存在，如果存在则添加后缀
            while os.path.exists(filename):
                filename = f"{filename_base}_{str(counter).zfill(7)}{file_extension}"
                counter += 1

                selected_colormap = self.colormap_var.get()  # 获取用户选择的颜色映射

            # 保存图像
            plt.imsave(filename, output, cmap=selected_colormap, origin="lower")
            print(f"图像已保存为: {filename}")
            print("保存完成时间")
            print(time.time())

            # 更新界面
            print("更新界面开始时间")
            print(time.time())
            metadata = {
                "x_range": (task["x_min"], task["x_max"]),
                "y_range": (task["y_min"], task["y_max"]),
                "task_id": task["task_id"],
                "compute_time": compute_time,
                "precision": task["precision"],
            }
            self.current_image = output
            self.current_metadata = metadata

            # 更新界面使用after方法
            self.root.after(0, self.update_canvas, output, metadata)

            print("更新完界面时间")
            print(time.time())
        # 清理状态
        generating = False
        self.status_var.set("生成完成" if current_task == total_tasks else "已终止")
        self.generate_button.config(text="开始生成", state="normal")
        self.toggle_inputs(True)
        self.toggle_julia_mode()

    def cleanup_after_error(self):
        """出错后清理"""
        global generating
        generating = False
        if hasattr(self, "generation_thread"):
            self.generation_thread.join(0)
        self.generate_button.config(text="开始生成", state="normal")
        self.status_var.set("就绪")
        self.toggle_inputs(True)
        self.toggle_julia_mode()


if __name__ == "__main__":

    root = Tk()
    app = MandelbrotGenerator(root)

    def on_close():
        global generating  # 明确声明使用全局变量
        generating = False
        root.destroy()
        # 强制退出进程（针对某些终端环境）
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)  # 使用自定义的关闭函数
    root.mainloop()
