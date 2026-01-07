import cv2
import numpy as np
import os

# 加载标定参数 - 不需要使用 .item()
calib_params = np.load("camera_calib_params.npz", allow_pickle=True)

# 查看npz文件中包含哪些键
print("npz文件中的键：", list(calib_params.keys()))

# 根据实际的键名访问数据
# 如果标定文件是用我的标定代码生成的，键名应该是：K, D, image_size, ...
# 如果标定文件是用您的标定代码生成的，键名可能是：camera_matrix, dist_coeffs, ...

# 方法1：尝试不同的键名组合
if 'K' in calib_params and 'D' in calib_params:
    mtx = calib_params["K"]
    dist = calib_params["D"]
    print("使用 K 和 D 键加载参数")
elif 'camera_matrix' in calib_params and 'dist_coeffs' in calib_params:
    mtx = calib_params["camera_matrix"]
    dist = calib_params["dist_coeffs"]
    print("使用 camera_matrix 和 dist_coeffs 键加载参数")
elif 'mtx' in calib_params and 'dist' in calib_params:
    mtx = calib_params["mtx"]
    dist = calib_params["dist"]
    print("使用 mtx 和 dist 键加载参数")
else:
    print("错误：未找到相机矩阵和畸变系数")
    print("可用键：", list(calib_params.keys()))
    exit()

# 获取图像尺寸
if 'image_size' in calib_params:
    img_size = calib_params["image_size"]
    print(f"图像尺寸：{img_size}")
else:
    # 如果没有保存图像尺寸，可以使用测试图像的尺寸
    img = cv2.imread("IMG_0001.jpg")
    if img is not None:
        h, w = img.shape[:2]
        img_size = (w, h)
        print(f"从测试图像获取尺寸：{img_size}")
    else:
        print("错误：无法获取图像尺寸")
        exit()

print(f"相机矩阵 K:\n{mtx}")
print(f"畸变系数 D: {dist.flatten()}")

# 读取测试图像（用一张包含目标的广角图像）
img = cv2.imread("IMG_0001.jpg")
if img is None:
    print("错误：无法读取测试图像 IMG_0001.jpg")
    exit()

h, w = img.shape[:2]

# 生成优化后的内参矩阵（避免校正后图像裁剪过多）
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 校正畸变
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# 裁剪校正后的图像（去除黑边，可选）
if roi is not None:
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y+h_roi, x:x+w_roi]
else:
    dst_cropped = dst
    x, y, w_roi, h_roi = 0, 0, w, h

# 显示对比
cv2.imshow("1. Original Image", img)
cv2.imshow("2. Undistorted Image", dst)
cv2.imshow("3. Undistorted + Cropped Image", dst_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存校正后的图像
# cv2.imwrite("undistorted_test_img.jpg", dst_cropped)
# print(f"校正后的图像已保存为 undistorted_test_img.jpg")
print(f"ROI区域：x={x}, y={y}, w={w_roi}, h={h_roi}")