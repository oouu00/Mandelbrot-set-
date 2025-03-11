#include <windows.h>
#include <math.h>
#include <gmp.h>
#include <mpfr.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // 添加时间头文件

// 定义比较函数
int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);  // 升序排序
}

// 定义 ThreadParams 结构体
typedef struct {
    int startRow;
    int endRow;
    BYTE* pBits;
    mpfr_t* mpfrVars;
    int iXmax;
    int iYmax;
    int IterationMax;
} ThreadParams;

int min_iter = 0;  // 最小迭代次数
double percentile = 0.98;  // 默认值为 0.98

// 默认参数
int NUM_THREADS = 4;  // 线程数量
int precision = 256;  // GMP/MPFR 精度（位数）
char* center_x = "-0.74515135254171761267617";  // 中心点 x 坐标
char* center_y = "0.13019734207256549877521";  // 中心点 y 坐标
char* width = "600";   // 图片宽度
char* height = "400";  // 图片高度
char* scale = "1.25829116006878627409549e-15";     // 当前缩放比例
char* focus_x = "0.53665115566"; // 焦点 x 坐标
char* focus_y = "0.532158525855555"; // 焦点 y 坐标
char* max_iter = "800"; // 最大迭代次数
char* zhuliyax = "0.0"; // 新增参数
char* zhuliyay = "0.0"; // 新增参数
char* zhuliya = "0";    // 新增参数
const double EscapeRadius = 2;  // 逃逸半径

int* iterationArray;  // 全局变量，迭代次数数组

// 迭代写入二进制文件
void SaveIterationArrayAsBinary(const char* filename, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    size_t bytesWritten = fwrite(iterationArray, sizeof(int), width * height, file);
    if (bytesWritten != (size_t)(width * height)) {
        printf("Failed to write all data to file: %s\n", filename);
    } else {
        printf("Successfully saved iteration array to %s\n", filename);
    }

    fclose(file);
}


// 处理 iterationArray 的函数
void ProcessIterationArray(int* array, int width, int height, int newMaxIter) {
    // 将数组中大于 newMaxIter 的值替换为 newMaxIter
    for (int iY = 0; iY < height; iY++) {
        for (int iX = 0; iX < width; iX++) {
            if (array[iY * width + iX] > newMaxIter) {
                array[iY * width + iX] = newMaxIter;
            }
        }
    }

    // 计算最小值 a 和最大值 max（忽略 -1）
    int a, max;
    int validCount = 0;
    int sum = 0;

    for (int iY = 0; iY < height; iY++) {
        for (int iX = 0; iX < width; iX++) {
            int value = array[iY * width + iX];
            if (value != -1) {
                sum += value;
                validCount++;
            }
        }
    }

    if (validCount == 0) {
        printf("No valid values in iterationArray\n");
        return;
    }

    a = newMaxIter;  // 初始化为最大值
    max = 0;

    for (int iY = 0; iY < height; iY++) {
        for (int iX = 0; iX < width; iX++) {
            int value = array[iY * width + iX];
            if (value != -1) {
                if (value < a) {
                    a = value;
                }
                if (value > max) {
                    max = value;
                }
            }
        }
    }

    // 计算 b
    int b = a + (max - a) * 0.2;

    // 对于每个 -1，检查其 3 格距离内的值是否都在 [a, b] 范围内或者都等于最大值
    for (int iY = 0; iY < height; iY++) {
        for (int iX = 0; iX < width; iX++) {
            if (array[iY * width + iX] == -1) {
                int count = 0;
                int total = 0;
                int valid = 1;

                // 检查 3 格距离内的值
                for (int dy = -3; dy <= 3; dy++) {
                    for (int dx = -3; dx <= 3; dx++) {
                        int ny = iY + dy;
                        int nx = iX + dx;

                        // 检查是否越界
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            int value = array[ny * width + nx];
                            if (value != -1) {
                                if ((value < a || value > b) && value != newMaxIter) {
                                    valid = 0;
                                    break;
                                }
                                total += value;
                                count++;
                            }
                        }
                    }
                    if (!valid) break;
                }

                if (valid && count > 0) {
                    int average = total / count;
                    array[iY * width + iX] = average;
                }
            }
        }
    }
}






// 采样函数
int CalculateNewMaxIter(int iXmax, int iYmax, mpfr_t* mpfrVars, int max_iter) {
    // 每次采样跳10个像素
    int step = 10;

    // 计算实际采样点的数量
    int sampleX = (iXmax + step - 1) / step;  // 向上取整
    int sampleY = (iYmax + step - 1) / step;  // 向上取整
    int totalSamples = sampleX * sampleY;

    // 动态分配内存存储采样结果
    int* sampleIterations = (int*)malloc(totalSamples * sizeof(int));
    if (!sampleIterations) {
        printf("Failed to allocate memory for sampleIterations\n");
        exit(1);
    }

    mpfr_t Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2;
    mpfr_inits2(precision, Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);
    mpfr_set_d(ER2, EscapeRadius * EscapeRadius, MPFR_RNDN);

    int sampleIndex = 0;
    for (int iY = 0; iY < iYmax; iY += step) {  // 每次增加 step
        mpfr_set_ui(temp, iY, MPFR_RNDN);
        mpfr_mul(temp, temp, mpfrVars[5], MPFR_RNDN);  // pixel_height
        mpfr_add(Cyy, mpfrVars[2], temp, MPFR_RNDN);     // y_min + iY * pixel_height

        for (int iX = 0; iX < iXmax; iX += step) {
            mpfr_set(Cy, Cyy, MPFR_RNDN);
            mpfr_set_ui(temp, iX, MPFR_RNDN);
            mpfr_mul(temp, temp, mpfrVars[4], MPFR_RNDN);  // pixel_width
            mpfr_add(Cx, mpfrVars[0], temp, MPFR_RNDN);     // x_min + iX * pixel_width

            mpfr_set_d(Zx, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy, 0.0, MPFR_RNDN);
            mpfr_set_d(Zx2, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy2, 0.0, MPFR_RNDN);

            // 根据 zhuliya 的值决定是否改变 Cx 和 Cy 的值
            unsigned long zhuliya_val = mpfr_get_ui(mpfrVars[13], MPFR_RNDN); // 获取整数值.   
            if (zhuliya_val == 1) {
                mpfr_set(Zx, Cx, MPFR_RNDN);
                mpfr_set(Zy, Cy, MPFR_RNDN);
                mpfr_set_str(Cx, zhuliyax, 10, MPFR_RNDN);
                mpfr_set_str(Cy, zhuliyay, 10, MPFR_RNDN);
            }

            int Iteration = 0;
            mpfr_add(temp, Zx2, Zy2, MPFR_RNDN);  // temp = Zx2 + Zy2
            while (Iteration < max_iter && mpfr_cmp(temp, ER2) < 0) {
                mpfr_mul(temp, Zx, Zy, MPFR_RNDN);
                mpfr_mul(Zx2, Zx, Zx, MPFR_RNDN);
                mpfr_mul(Zy2, Zy, Zy, MPFR_RNDN);
                mpfr_mul_2exp(temp, temp, 1, MPFR_RNDN);  // temp = 2 * Zx * Zy
                mpfr_add(Zy, temp, Cy, MPFR_RNDN);

                mpfr_sub(temp, Zx2, Zy2, MPFR_RNDN);
                mpfr_add(Zx, temp, Cx, MPFR_RNDN);

                mpfr_add(temp, Zx2, Zy2, MPFR_RNDN);  // temp = Zx2 + Zy2
                Iteration++;
            }


            // 将采样结果存储到 iterationArray
            iterationArray[iY * iXmax + iX] = Iteration;
            sampleIterations[sampleIndex++] = Iteration;  // 保存迭代次数
        }
    }

    // 在采样循环结束后，找到最小迭代次数
    int minIteration = sampleIterations[0];
    for (int i = 0; i < totalSamples; i++) {
        if (sampleIterations[i] < minIteration) {
            minIteration = sampleIterations[i];
        }
    }
    min_iter = minIteration;  // 更新全局变量 min_iter
    printf("  Min Iterations: %d\n", min_iter);

    // 找到最大迭代次数并删除
    int maxIteration = 0;
    for (int i = 0; i < totalSamples; i++) {
        if (sampleIterations[i] > maxIteration) {
            maxIteration = sampleIterations[i];
        }
    }

    // 删除最大值
    int* filteredIterations = (int*)malloc((totalSamples - 1) * sizeof(int));
    if (!filteredIterations) {
        printf("Failed to allocate memory for filteredIterations\n");
        exit(1);
    }

    int filteredIndex = 0;
    for (int i = 0; i < totalSamples; i++) {
        if (sampleIterations[i] != maxIteration) {
            filteredIterations[filteredIndex++] = sampleIterations[i];
        }
    }

    // 使用过滤后的数组计算98%分位数
    int* sortedIterations = (int*)malloc(filteredIndex * sizeof(int));
    if (!sortedIterations) {
        printf("Failed to allocate memory for sortedIterations\n");
        exit(1);
    }
    memcpy(sortedIterations, filteredIterations, filteredIndex * sizeof(int));
    qsort(sortedIterations, filteredIndex, sizeof(int), compare);

    int newMaxIter = sortedIterations[(int)(percentile * filteredIndex)];
    
    

    // 将 newMaxIter 与 100 比较，取最大值
    newMaxIter = (newMaxIter > 100) ? newMaxIter : 100;

    printf("Sample Iterations Statistics:\n");
    printf("  Total Samples: %d\n", totalSamples);
    printf("  Min Iterations: %d\n", sortedIterations[0]);
    printf("  Max Iterations: %d\n", sortedIterations[filteredIndex - 1]);
    printf("  98%% Percentile Iterations: %d\n", newMaxIter);

    free(sampleIterations);
    free(sortedIterations);
    free(filteredIterations);

    mpfr_clears(Cx, Cy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);

    return newMaxIter;
}


//正式前采样后预先计算

unsigned __stdcall MandelbrotThread2(void* param) {
    ThreadParams* params = (ThreadParams*)param;
    int startRow = params->startRow;
    int endRow = params->endRow;
    BYTE* pBits = params->pBits;
    mpfr_t* mpfrVars = params->mpfrVars;
    int iXmax = params->iXmax;
    int iYmax = params->iYmax;
    int IterationMax = params->IterationMax;

    mpfr_t Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2;
    mpfr_inits2(precision, Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);

    mpfr_set_d(ER2, EscapeRadius * EscapeRadius, MPFR_RNDN);

    // 隔3行采样
    for (int iY = startRow; iY < endRow; iY += 3) {
        mpfr_set_ui(temp, iY, MPFR_RNDN);
        mpfr_mul(temp, temp, mpfrVars[5], MPFR_RNDN);  // pixel_height
        mpfr_add(Cyy, mpfrVars[2], temp, MPFR_RNDN);     // y_min + iY * pixel_height

        // 隔3格采样
        for (int iX = 0; iX < iXmax; iX += 3) {
            // 如果 iterationArray 中已经有值（不是 NaN），则跳过计算
            if (iterationArray[iY * iXmax + iX] != -1) {
                continue;
            }

            mpfr_set(Cy, Cyy, MPFR_RNDN);
            mpfr_set_ui(temp, iX, MPFR_RNDN);
            mpfr_mul(temp, temp, mpfrVars[4], MPFR_RNDN);  // pixel_width
            mpfr_add(Cx, mpfrVars[0], temp, MPFR_RNDN);     // x_min + iX * pixel_width

            mpfr_set_d(Zx, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy, 0.0, MPFR_RNDN);
            mpfr_set_d(Zx2, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy2, 0.0, MPFR_RNDN);

            // 根据 zhuliya 的值决定是否改变 Cx 和 Cy 的值
            unsigned long zhuliya_val = mpfr_get_ui(mpfrVars[13], MPFR_RNDN); // 获取整数值.   
            if (zhuliya_val == 1) {
                mpfr_set(Zx, Cx, MPFR_RNDN);
                mpfr_set(Zy, Cy, MPFR_RNDN);
                mpfr_set_str(Cx, zhuliyax, 10, MPFR_RNDN);
                mpfr_set_str(Cy, zhuliyay, 10, MPFR_RNDN);
            }

            int Iteration = 0;
            while (Iteration < IterationMax) {
                mpfr_mul(temp, Zx, Zy, MPFR_RNDN);
                mpfr_mul(Zx2, Zx, Zx, MPFR_RNDN);
                mpfr_mul(Zy2, Zy, Zy, MPFR_RNDN);
                mpfr_mul_2exp(temp, temp, 1, MPFR_RNDN);  // temp = 2 * Zx * Zy
                mpfr_add(Zy, temp, Cy, MPFR_RNDN);

                mpfr_sub(temp, Zx2, Zy2, MPFR_RNDN);
                mpfr_add(Zx, temp, Cx, MPFR_RNDN);

                // 只有在迭代次数达到 min_iter 次之后才开始判断是否逃逸
                if (Iteration >= min_iter) {
                    mpfr_add(temp, Zx2, Zy2, MPFR_RNDN);  // temp = Zx2 + Zy2
                    if (mpfr_cmp(temp, ER2) >= 0) {  // 如果逃逸，退出循环
                        break;
                    }
                }
                Iteration++;
            }
            
            // 迭代次数平滑
            double result = (Iteration < IterationMax) 
                ? Iteration + 1 - log(log(sqrt(mpfr_get_d(Zx2, MPFR_RNDN) + mpfr_get_d(Zy2, MPFR_RNDN))) / log(2.0)) / log(2.0)
                : IterationMax;

            iterationArray[iY * iXmax + iX] = result;  // 保存迭代次数
        }
    }

    mpfr_clears(Cx, Cy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);
    _endthreadex(0);
    return 0;
}







// 线程函数
unsigned __stdcall MandelbrotThread(void* param) {
    ThreadParams* params = (ThreadParams*)param;
    int startRow = params->startRow;
    int endRow = params->endRow;
    BYTE* pBits = params->pBits;
    mpfr_t* mpfrVars = params->mpfrVars;
    int iXmax = params->iXmax;
    int iYmax = params->iYmax;
    int IterationMax = params->IterationMax;

    mpfr_t Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2;
    mpfr_inits2(precision, Cx, Cy, Cyy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);

    mpfr_set_d(ER2, EscapeRadius * EscapeRadius, MPFR_RNDN);

    for (int iY = startRow; iY < endRow; iY++) {
        mpfr_set_ui(temp, iY, MPFR_RNDN);
        mpfr_mul(temp, temp, mpfrVars[5], MPFR_RNDN);  // pixel_height
        mpfr_add(Cyy, mpfrVars[2], temp, MPFR_RNDN);     // y_min + iY * pixel_height

        for (int iX = 0; iX < iXmax; iX++) {
            // 如果 iterationArray 中已经有值（不是 NaN），则跳过计算
            if (iterationArray[iY * iXmax + iX] != -1) {
                continue;
            }

            mpfr_set(Cy, Cyy, MPFR_RNDN);
            mpfr_set_ui(temp, iX, MPFR_RNDN);
            mpfr_mul(temp, temp, mpfrVars[4], MPFR_RNDN);  // pixel_width
            mpfr_add(Cx, mpfrVars[0], temp, MPFR_RNDN);     // x_min + iX * pixel_width

            mpfr_set_d(Zx, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy, 0.0, MPFR_RNDN);
            mpfr_set_d(Zx2, 0.0, MPFR_RNDN);
            mpfr_set_d(Zy2, 0.0, MPFR_RNDN);

            // 根据 zhuliya 的值决定是否改变 Cx 和 Cy 的值
            unsigned long zhuliya_val = mpfr_get_ui(mpfrVars[13], MPFR_RNDN); // 获取整数值.   
            if (zhuliya_val == 1) {
                mpfr_set(Zx, Cx, MPFR_RNDN);
                mpfr_set(Zy, Cy, MPFR_RNDN);
                mpfr_set_str(Cx, zhuliyax, 10, MPFR_RNDN);
                mpfr_set_str(Cy, zhuliyay, 10, MPFR_RNDN);
            }

            int Iteration = 0;
            while (Iteration < IterationMax) {
                mpfr_mul(temp, Zx, Zy, MPFR_RNDN);
                mpfr_mul(Zx2, Zx, Zx, MPFR_RNDN);
                mpfr_mul(Zy2, Zy, Zy, MPFR_RNDN);
                mpfr_mul_2exp(temp, temp, 1, MPFR_RNDN);  // temp = 2 * Zx * Zy
                mpfr_add(Zy, temp, Cy, MPFR_RNDN);

                mpfr_sub(temp, Zx2, Zy2, MPFR_RNDN);
                mpfr_add(Zx, temp, Cx, MPFR_RNDN);

                // 只有在迭代次数达到 min_iter 次之后才开始判断是否逃逸
                if (Iteration >= min_iter) {
                    mpfr_add(temp, Zx2, Zy2, MPFR_RNDN);  // temp = Zx2 + Zy2
                    if (mpfr_cmp(temp, ER2) >= 0) {  // 如果逃逸，退出循环
                        break;
                    }
                }
                Iteration++;
            }
            
            // 迭代次数平滑
            double result = (Iteration < IterationMax) 
                ? Iteration + 1 - log(log(sqrt(mpfr_get_d(Zx2, MPFR_RNDN) + mpfr_get_d(Zy2, MPFR_RNDN))) / log(2.0)) / log(2.0)
                : IterationMax;

            iterationArray[iY * iXmax + iX] = result;  // 保存迭代次数
        }
    }

    mpfr_clears(Cx, Cy, Zx, Zy, Zx2, Zy2, temp, ER2, (mpfr_ptr)0);
    _endthreadex(0);
    return 0;
}

// 读取文件函数
void ParseParametersFromFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Failed to open parameter file: %s\n", filename);
        exit(1);
    }

    char line[1048576];  // 假设每行的最大长度为1048576个字符

    // 动态分配内存以存储长字符串
    center_x = (char*)malloc(1048576);
    center_y = (char*)malloc(1048576);
    width = (char*)malloc(1048576);
    height = (char*)malloc(1048576);
    scale = (char*)malloc(1048576);
    focus_x = (char*)malloc(1048576);
    focus_y = (char*)malloc(1048576);
    max_iter = (char*)malloc(1048576);
    zhuliyax = (char*)malloc(1048576);
    zhuliyay = (char*)malloc(1048576);
    zhuliya = (char*)malloc(1048576);

    // 逐行读取参数
    fgets(line, sizeof(line), file);  // 读取线程数量
    sscanf(line, "%d", &NUM_THREADS);

    fgets(line, sizeof(line), file);  // 读取精度
    sscanf(line, "%d", &precision);

    fgets(line, sizeof(line), file);  // 读取中心点 X 坐标
    sscanf(line, "%s", center_x);

    fgets(line, sizeof(line), file);  // 读取中心点 Y 坐标
    sscanf(line, "%s", center_y);

    fgets(line, sizeof(line), file);  // 读取宽度
    sscanf(line, "%s", width);

    fgets(line, sizeof(line), file);  // 读取高度
    sscanf(line, "%s", height);

    fgets(line, sizeof(line), file);  // 读取缩放比例
    sscanf(line, "%s", scale);

    fgets(line, sizeof(line), file);  // 读取焦点 X 坐标
    sscanf(line, "%s", focus_x);

    fgets(line, sizeof(line), file);  // 读取焦点 Y 坐标
    sscanf(line, "%s", focus_y);

    fgets(line, sizeof(line), file);  // 读取最大迭代次数
    sscanf(line, "%s", max_iter);

    fgets(line, sizeof(line), file);  // 读取百分位数
    if (sscanf(line, "%lf", &percentile) != 1) {
        printf("Percentile not specified in file, using default value: 0.98\n");
        percentile = 0.98;  // 如果读取失败，使用默认值
    } else {
        printf("Percentile specified in file: %.2f\n", percentile);
    }

    fgets(line, sizeof(line), file);  // 读取 zhuliyax
    sscanf(line, "%s", zhuliyax);

    fgets(line, sizeof(line), file);  // 读取 zhuliyay
    sscanf(line, "%s", zhuliyay);

    fgets(line, sizeof(line), file);  // 读取 zhuliya
    sscanf(line, "%s", zhuliya);

    fclose(file);

    // 打印读取到的参数
    printf("Using parameters:\n");
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Precision: %d bits\n", precision);
    printf("  Center: (%s, %s)\n", center_x, center_y);
    printf("  Width: %s, Height: %s\n", width, height);
    printf("  Scale: %s\n", scale);
    printf("  Focus: (%s, %s)\n", focus_x, focus_y);
    printf("  Max Iterations: %s\n", max_iter);
    printf("  Percentile: %.2f\n", percentile);
    printf("  zhuliyax: %s\n", zhuliyax);
    printf("  zhuliyay: %s\n", zhuliyay);
    printf("  zhuliya: %s\n", zhuliya);
}

// 主函数
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: Mandelbrot.exe <parameter_file>\n");
        return -1;
    }

    const char* paramFile = argv[1];

    ParseParametersFromFile(paramFile);  // 从文件读取参数

    // 初始化 GMP/MPFR 精度
    mpfr_set_default_prec(precision);

    // 解析输入参数
    mpfr_t center_x_mp, center_y_mp, scale_mp, focus_x_mp, focus_y_mp;
    mpfr_inits2(precision, center_x_mp, center_y_mp, scale_mp, focus_x_mp, focus_y_mp, (mpfr_ptr)0);

    mpfr_set_str(center_x_mp, center_x, 10, MPFR_RNDN);  // 中心点 X 坐标
    mpfr_set_str(center_y_mp, center_y, 10, MPFR_RNDN);  // 中心点 Y 坐标
    mpfr_set_str(scale_mp, scale, 10, MPFR_RNDN);        // 缩放比例
    mpfr_set_str(focus_x_mp, focus_x, 10, MPFR_RNDN);    // 焦点 X 坐标
    mpfr_set_str(focus_y_mp, focus_y, 10, MPFR_RNDN);    // 焦点 Y 坐标

    int iXmax = atoi(width);    // 图片宽度
    int iYmax = atoi(height);   // 图片高度
    int IterationMax = atoi(max_iter);  // 最大迭代次数

    // 为全局变量 iterationArray 分配内存，并初始化为 -1（模拟 NaN）
    iterationArray = (int*)malloc(iYmax * iXmax * sizeof(int));
    if (!iterationArray) {
        printf("Failed to allocate memory for iterationArray\n");
        return -1;
    }
    memset(iterationArray, -1, iYmax * iXmax * sizeof(int));  // 初始化为 -1

    // 计算像素宽度和高度
    mpfr_t pixel_width, pixel_height, iXmax_mp, iYmax_mp;
    mpfr_inits(pixel_width, pixel_height, iXmax_mp, iYmax_mp, (mpfr_ptr)0);
    mpfr_set_ui(iXmax_mp, iXmax, MPFR_RNDN);
    mpfr_set_ui(iYmax_mp, iYmax, MPFR_RNDN);

    mpfr_t x_min, x_max, y_min, y_max, x_range, y_range;
    mpfr_inits(x_min, x_max, y_min, y_max, x_range, y_range, (mpfr_ptr)0);

    mpfr_t temp, mp_one, mp_two;
    mpfr_inits(temp, mp_one, mp_two, (mpfr_ptr)0);

    mpfr_set_ui(mp_one, 1, MPFR_RNDN);
    mpfr_set_ui(mp_two, 2, MPFR_RNDN);

    // 计算复平面的 X 范围
    mpfr_mul(temp, scale_mp, focus_x_mp, MPFR_RNDN);
    mpfr_sub(x_min, center_x_mp, temp, MPFR_RNDN);
    mpfr_sub(temp, scale_mp, temp, MPFR_RNDN);
    mpfr_add(x_max, center_x_mp, temp, MPFR_RNDN);

    // 计算复平面的 Y 范围
    mpfr_div(temp, iYmax_mp, iXmax_mp, MPFR_RNDN);
    mpfr_mul(temp, temp, focus_y_mp, MPFR_RNDN);
    mpfr_mul(temp, scale_mp, temp, MPFR_RNDN);
    mpfr_sub(y_min, center_y_mp, temp, MPFR_RNDN);

    mpfr_sub(temp, mp_one, focus_y_mp, MPFR_RNDN);
    mpfr_mul(temp, temp, scale_mp, MPFR_RNDN);
    mpfr_div(mp_one, iYmax_mp, iXmax_mp, MPFR_RNDN);
    mpfr_mul(temp, temp, mp_one, MPFR_RNDN);
    mpfr_add(y_max, center_y_mp, temp, MPFR_RNDN);

    mpfr_clears(temp, mp_one, mp_two, (mpfr_ptr)0);

    // 计算复平面的 X 和 Y 范围
    mpfr_sub(x_range, x_max, x_min, MPFR_RNDN);
    mpfr_sub(y_range, y_max, y_min, MPFR_RNDN);

    // 计算每个像素对应的复平面宽度和高度
    mpfr_div(pixel_width, x_range, iXmax_mp, MPFR_RNDN);
    mpfr_div(pixel_height, y_range, iYmax_mp, MPFR_RNDN);

    // 初始化线程参数
    ThreadParams threadParams[NUM_THREADS];
    mpfr_t* mpfrVars = (mpfr_t*)malloc(14 * sizeof(mpfr_t));  // 增加一个变量用于存储 zhuliya
    for (int i = 0; i < 14; i++) {
        mpfr_init(mpfrVars[i]);
    }

    mpfr_set(mpfrVars[0], x_min, MPFR_RNDN);
    mpfr_set(mpfrVars[1], x_max, MPFR_RNDN);
    mpfr_set(mpfrVars[2], y_min, MPFR_RNDN);
    mpfr_set(mpfrVars[3], y_max, MPFR_RNDN);
    mpfr_set(mpfrVars[4], pixel_width, MPFR_RNDN);
    mpfr_set(mpfrVars[5], pixel_height, MPFR_RNDN);
    mpfr_set(mpfrVars[6], center_x_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[7], center_y_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[8], scale_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[9], focus_x_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[10], focus_y_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[11], iXmax_mp, MPFR_RNDN);
    mpfr_set(mpfrVars[12], iYmax_mp, MPFR_RNDN);
    mpfr_set_str(mpfrVars[13], zhuliya, 10, MPFR_RNDN);  // 存储 zhuliya 的值

    // 计算采样点的迭代次数并确定新的最大迭代次数
    int newMaxIter = CalculateNewMaxIter(iXmax, iYmax, mpfrVars, IterationMax);
    printf("New Max Iterations based on sampling: %d\n", newMaxIter);
    
        // 使用新的最大迭代次数重新计算整个图像
    int rowsPerThread = iYmax / NUM_THREADS;
    HANDLE hThreads2[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? iYmax : startRow + rowsPerThread;

        threadParams[i].startRow = startRow;
        threadParams[i].endRow = endRow;
        threadParams[i].pBits = NULL;  // 不使用位图，设置为 NULL
        threadParams[i].mpfrVars = mpfrVars;
        threadParams[i].iXmax = iXmax;
        threadParams[i].iYmax = iYmax;
        threadParams[i].IterationMax = newMaxIter;

        hThreads2[i] = (HANDLE)_beginthreadex(NULL, 0, MandelbrotThread2, (void*)&threadParams[i], 0, NULL);
    }

    // 等待线程完成
    WaitForMultipleObjects(NUM_THREADS, hThreads2, TRUE, INFINITE);
    for (int i = 0; i < NUM_THREADS; i++) {
        CloseHandle(hThreads2[i]);
    }
    
    
    

    // 在调用 ProcessIterationArray 函数前记录开始时间
    time_t start_time, end_time;
    time(&start_time);
    printf("Processing iterationArray started at %s", ctime(&start_time));

    // 调用 ProcessIterationArray 函数
    ProcessIterationArray(iterationArray, iXmax, iYmax, newMaxIter);

    // 在调用 ProcessIterationArray 函数后记录结束时间
    time(&end_time);
    printf("Processing iterationArray finished at %s", ctime(&end_time));


    // 使用新的最大迭代次数重新计算整个图像
    HANDLE hThreads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? iYmax : startRow + rowsPerThread;

        threadParams[i].startRow = startRow;
        threadParams[i].endRow = endRow;
        threadParams[i].pBits = NULL;  // 不使用位图，设置为 NULL
        threadParams[i].mpfrVars = mpfrVars;
        threadParams[i].iXmax = iXmax;
        threadParams[i].iYmax = iYmax;
        threadParams[i].IterationMax = newMaxIter;

        hThreads[i] = (HANDLE)_beginthreadex(NULL, 0, MandelbrotThread, (void*)&threadParams[i], 0, NULL);
    }

    // 等待线程完成
    WaitForMultipleObjects(NUM_THREADS, hThreads, TRUE, INFINITE);
    for (int i = 0; i < NUM_THREADS; i++) {
        CloseHandle(hThreads[i]);
    }

    // 清理 MPFR 变量
    for (int i = 0; i < 14; i++) {
        mpfr_clear(mpfrVars[i]);
    }
    free(mpfrVars);

    // 保存迭代次数数组
    SaveIterationArrayAsBinary("iteration_array.bin", iXmax, iYmax);

    // 释放全局变量 iterationArray
    free(iterationArray);

    // 清理 MPFR 变量
    mpfr_clears(center_x_mp, center_y_mp, scale_mp, focus_x_mp, focus_y_mp,
                pixel_width, pixel_height, iXmax_mp, iYmax_mp,
                x_min, x_max, y_min, y_max, x_range, y_range, (mpfr_ptr)0);

    return 0;
}
