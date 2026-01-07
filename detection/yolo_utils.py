"""
YOLO检测核心逻辑 - 检测框解析、目标筛选
"""
from config import settings

def extract_box_info(box, model, scale=None):
    """提取检测框信息"""
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id] if cls_id < len(model.names) else f"未知类别({cls_id})"
    conf = float(box.conf[0])
    # xyxy may be a tensor; convert to floats
    try:
        coords = box.xyxy[0].cpu().numpy()
    except Exception:
        coords = box.xyxy[0]
    x1_f, y1_f, x2_f, y2_f = map(float, coords)
    if scale is not None:
        sx, sy = scale
        x1 = int(round(x1_f * sx))
        y1 = int(round(y1_f * sy))
        x2 = int(round(x2_f * sx))
        y2 = int(round(y2_f * sy))
    else:
        x1, y1, x2, y2 = map(int, (x1_f, y1_f, x2_f, y2_f))
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    box_width = x2 - x1
    box_height = y2 - y1

    return {
        "cls_id": cls_id,
        "cls_name": cls_name,
        "conf": conf,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": center_x,
        "center_y": center_y,
        "width": box_width,
        "height": box_height
    }

def get_first_detected_target(results, model, frame_id, scale=None):
    """提取第一个检测到的目标"""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, []
    
    # 打印所有检测到的目标
    all_targets = []
    print(f"\n【第 {frame_id} 帧】所有检测结果：")
    for i, box in enumerate(boxes):
        target_info = extract_box_info(box, model, scale=scale)
        all_targets.append(target_info)
        print(f"  目标{i+1}：类别={target_info['cls_name']}(ID={target_info['cls_id']}) | 置信度={target_info['conf']:.4f}")
    
    # 返回第一个置信度达标的目标
    first_target = None
    for target in all_targets:
        if target["conf"] >= settings.CONF_THRESHOLD:
            first_target = target
            break
    return first_target, all_targets