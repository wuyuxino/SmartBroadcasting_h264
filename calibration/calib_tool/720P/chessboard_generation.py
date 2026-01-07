import cv2
import numpy as np
import sys
import os
from PIL import Image

# ===================== 核心参数（适配1280×720） =====================
A4_WIDTH_MM = 210.0       # A4纸宽度（mm）
A4_HEIGHT_MM = 297.0      # A4纸高度（mm）
GRID_SIZE_MM = 21.05       # 格子物理尺寸（mm），打印后若为19mm可改为21.05
PRINT_DPI = 600           # 打印机DPI
MM_PER_INCH = 25.4        # 1英寸=25.4mm
MARGIN_MM = 10.0          # 预留打印页边距
GRID_COLS = 9             # 格子列数→8个内角点（列）
GRID_ROWS = 5             # 格子行数→4个内角点（行）

# ===================== 精准计算（含页边距） =====================
effective_width_mm = A4_WIDTH_MM - 2 * MARGIN_MM
effective_height_mm = A4_HEIGHT_MM - 2 * MARGIN_MM

a4_width_px = A4_WIDTH_MM / MM_PER_INCH * PRINT_DPI
a4_height_px = A4_HEIGHT_MM / MM_PER_INCH * PRINT_DPI
grid_size_px = GRID_SIZE_MM / MM_PER_INCH * PRINT_DPI

total_grid_width_px = GRID_COLS * grid_size_px
total_grid_height_px = GRID_ROWS * grid_size_px

offset_x = MARGIN_MM / MM_PER_INCH * PRINT_DPI + (effective_width_mm / MM_PER_INCH * PRINT_DPI - total_grid_width_px) / 2
offset_y = MARGIN_MM / MM_PER_INCH * PRINT_DPI + (effective_height_mm / MM_PER_INCH * PRINT_DPI - total_grid_height_px) / 2

# 创建纯白背景
calib_board = np.ones((int(round(a4_height_px)), int(round(a4_width_px)), 3), dtype=np.uint8) * 255

# 绘制棋盘格
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        if (row + col) % 2 == 0:
            x1 = offset_x + col * grid_size_px
            y1 = offset_y + row * grid_size_px
            x2 = x1 + grid_size_px
            y2 = y1 + grid_size_px
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            calib_board[y1:y2, x1:x2] = (0, 0, 0)

# ===================== 保存（带DPI元数据） =====================
script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
save_path = os.path.join(script_dir, "720P_20mm_8x4_corners_600DPI.png")
cv2.imwrite(save_path, calib_board)
# 补充DPI元数据（避免打印缩放）
with Image.open(save_path) as img:
    img.save(save_path, dpi=(PRINT_DPI, PRINT_DPI))

# ===================== 输出关键信息 =====================
print("✅ 适配1280×720的标定板生成完成！")
print(f"   格子物理尺寸：{GRID_SIZE_MM}mm → 像素尺寸：{grid_size_px:.2f}×{grid_size_px:.2f}px")
print(f"   内角点数量：{GRID_COLS-1}×{GRID_ROWS-1}（列×行）")
print(f"   打印建议：600DPI、无缩放、无边距打印")
print(f"   保存路径：{save_path}")

# 缩放显示（避免窗口过大）
display_board = cv2.resize(calib_board, (800, 1130))
cv2.imshow("720P Calibration Board (8x4 Corners)", display_board)
cv2.waitKey(0)
cv2.destroyAllWindows()