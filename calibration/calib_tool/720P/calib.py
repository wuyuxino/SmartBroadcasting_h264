import cv2
import numpy as np
import glob
import os

# ===================== 核心参数（根据实际标定板修改！） =====================
# 🔴 修改1：匹配标定板内角点（列×行）
# 720P（1280×720）用(8,4)，1080P（1920×1080）用(9,5)
board_size = (8, 4)        
# 🔴 修改2：补偿后的格子尺寸（打印后≈20mm）
square_size = 21.05         

script_dir = os.path.dirname(os.path.abspath(__file__))
calib_img_path = os.path.join(script_dir, "calib_images", "*.png")  
save_params_path = os.path.join(script_dir, "camera_calib_params.npz")  

# ===================== 初始化变量 =====================
obj_points = []  
img_points = []  
img_size = None  
# 🔴 新增：记录所有图的尺寸，用于校验一致性
all_img_sizes = []  

# 生成棋盘格的3D角点坐标
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ===================== 遍历标定图像，检测角点 =====================
img_paths = glob.glob(calib_img_path)
if len(img_paths) == 0:
    print(f"错误：未找到标定图像！请检查路径：{calib_img_path}")
    exit()

print(f"找到{len(img_paths)}张标定图像，开始检测角点...")

for img_path in img_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：跳过无效图像 {img_path}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    curr_img_size = gray.shape[::-1]  # (宽, 高)
    all_img_sizes.append(curr_img_size)
    
    # 🔴 修改3：校验所有图像尺寸一致
    if img_size is None:
        img_size = curr_img_size
    else:
        if curr_img_size != img_size:
            print(f"警告：{os.path.basename(img_path)}尺寸{curr_img_size}≠{img_size}，跳过！")
            continue
    
    # 检测角点（新增参数提升检测成功率）
    ret, corners = cv2.findChessboardCorners(
        gray, board_size, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)
        print(f"✅ {os.path.basename(img_path)}：检测到{len(corners)}个角点")
    else:
        print(f"❌ {os.path.basename(img_path)}：未检测到角点，跳过")

if len(obj_points) == 0:
    print("错误：无有效角点数据，无法标定！")
    exit()

# ===================== 执行相机标定 + 计算最优内参（核心修改） =====================
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, img_size, None, None
)

# 🔴 修改4：计算最优新内参+ROI（解决过度裁剪）
# alpha=0.4：平衡畸变和裁剪幅度，0=仅保留无畸变区（裁剪最大），1=保留全画面（畸变最大）
alpha = 0.4
new_mtx, roi = cv2.getOptimalNewCameraMatrix(
    mtx, dist, img_size, alpha=alpha, centerPrincipalPoint=True
)

# ===================== 计算重投影误差 + 过滤低质量图 =====================
mean_error = 0
bad_img_indices = []
# 先计算所有图的误差
for i in range(len(obj_points)):
    img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
    mean_error += error
    # 过滤重投影误差＞1像素的低质量图
    if error > 1.0:
        bad_img_indices.append(i)
        print(f"警告：第{i}张图重投影误差{error:.4f}＞1，已过滤")

# 若有低质量图，重新标定
if bad_img_indices:
    obj_points = [p for i, p in enumerate(obj_points) if i not in bad_img_indices]
    img_points = [p for i, p in enumerate(img_points) if i not in bad_img_indices]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    # 重新计算最优内参+ROI
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, img_size, alpha=alpha, centerPrincipalPoint=True
    )
    # 重新计算误差
    mean_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        mean_error += error

mean_error /= len(obj_points) if len(obj_points) > 0 else 1

# ===================== 保存标定参数（新增new_K和roi） =====================
np.savez(
    save_params_path,
    K=mtx,          # 原始内参
    new_K=new_mtx,  # 🔴 修改5：保存最优内参（裁剪用）
    D=dist,
    roi=roi,        # 🔴 修改5：保存适配的ROI（裁剪用）
    image_size=img_size,
    mean_error=mean_error,
    board_size=board_size,
    square_size=square_size,
    alpha=alpha     # 保存alpha参数，方便后续调整
)

# ===================== 输出标定结果 =====================
print("\n" + "="*60)
print("✅ 相机标定完成（适配720P/1080P）！")
print(f"📷 原始内参矩阵（K）:\n{mtx}")
print(f"📷 最优内参矩阵（new_K）:\n{new_mtx}")  # 裁剪代码用这个！
print(f"📏 畸变系数（D）:\n{dist}")
print(f"🎯 重投影误差（目标＜1）: {mean_error:.4f} 像素")
print(f"🖼️  图像尺寸: {img_size[0]}×{img_size[1]}")
print(f"✂️  推荐ROI（x,y,w,h）: {roi}")  # 裁剪代码用这个！
print(f"💾 标定参数已保存至: {save_params_path}")
print("="*60)

# 测试加载参数
print("\n测试加载标定参数...")
try:
    calib_data = np.load(save_params_path)
    print(f"✅ 参数文件加载成功，包含的键：{list(calib_data.keys())}")
    print(f"   new_K的形状：{calib_data['new_K'].shape}")
    print(f"   roi：{calib_data['roi']}")
except Exception as e:
    print(f"❌ 加载失败：{e}")