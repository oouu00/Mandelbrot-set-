# Mandelbrot-set-
py+c写的曼德博集图片生成器，也可以生成朱丽叶集的图片,可以无限放大，无限精度，在浮点数计算范围内GPU运算很快，后面高精度了使用gmp计算就很慢了,采样了平滑区域跳过计算了还是挺慢的

<img width="601" alt="1" src="https://github.com/user-attachments/assets/bb77cd2f-9b1a-4bda-904b-593ddbb271fd" />



参数说明:
1.分辨率就是图片输出分辨率，不是窗口显示的，窗口显示的是缩放的
2.曼德博集方程 Zₐ₊₁ = Zₐ² + c,xy是复数c的实部和虚部，就是图片生成时的复平面坐标，下面6个按钮就预先设置的xy坐标，就第一个设定的精细，可以直接放大10^50
3.宽度数值就是图片复平面从左到右的轴长度数值，越小图片放大越大
4.最大迭代 9999，一般可以不用改，采样算到这个直了停止来算新的实际迭代次数
5.迭代次数的阈值 0-100之间，越高自动迭代次数越高
6.生成数量就是一次生成的图片张数，多次生成图片会后续继续编号保存，不覆盖之前图片
7.放大系数就是第二张图片比第一张放大的倍数，可以小于1来缩小，必须大于0
8.焦点位置坐标xy 这个就是焦点数值xy坐标这个点所在图片中的位置，0，0是左下角，1，1是右上角
9.切换cpu的精度阈值，默认11  就是图片对应的复平面宽度小于10^-11就切换cpu
10.颜色映射 切换颜色模式，不同颜色映射
11.保存路径和开始终止
12.右边图片点击可以改放大中心，拖动可以变这个中心的位置(就是把图片拖过来再放大)
13.图片上当前精度超出8那里的阈值，就切换高精度计算，就是用gmp计算模式，其他像OpenCL等模式很快的
14.图片上的计算时间只是图片算出来的时间，其他显示和保存图片的步骤没算进去
15.启用茱莉亚集后可以输入C的XY坐标.


c代码编译为mandelbrot.exe放py代码同目录，然后py运行就好,py脚本运算会调用这个exe



这里的mandelbrot.exe是实现接收配置文件，cmd运行了生成包含迭代次数的二维数组.bin的功能
配置文件是一个txt文本文件，其中包含程序运行所需的参数。每行一个参数，顺序如下：1. 线程数量2. GMP/MPFR 精度（位数）3. 中心点 X 坐标4. 中心点 Y 坐标5. 图片宽度6. 图片高度7. 当前缩放比例8. 焦点 X 坐标9. 焦点 Y 坐标10. 最大迭代次数11. 自动迭代阈值（0-1小数),12. 朱丽亚集C的x左边;13.朱丽亚集C的y坐标,14.是否启用朱丽亚集






