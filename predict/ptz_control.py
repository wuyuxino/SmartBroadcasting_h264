import serial
import time
import threading
import numpy as np
from typing import Optional, Tuple

# ---------------------- 基础配置常量 ----------------------
# 串口配置
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
CONTROL_TIMEOUT = 0.1  # 控制指令超时（高频调用）
QUERY_TIMEOUT = 0.5    # 查询指令超时（需要更长等待）
# 云台角度配置
ANGLE_COEFF = 0.075
DEFAULT_SPEED = 0x08
ANGLE_RANGE_H = (-168, 168)  # 水平角度范围
ANGLE_RANGE_V = (-30, 90)    # 垂直角度范围
# 线程锁（多线程调用安全）
PTZ_LOCK = threading.Lock()

# ---------------------- 变焦核心参数（你的实测值） ----------------------
FOCAL_WIDE = 7.1               # 广角端焦距（1倍）
FOCAL_TELE_NOM = 7.1 * 25      # 远端焦距（25倍）
ZOOM_RAW_MAX = 16384            # 变焦raw值最大值（对应25倍）
ZOOM_RANGE = (1.0, 25.0)       # 变焦倍数范围（1~25倍）
DIST_RANGE = (2.5, 8.0)        # 有效距离范围

# ---------------------- 足球/变焦映射参数（拟合结果） ----------------------
# 1. 像素直径→距离 拟合参数
PIXEL2DIST_K = 107.43          # 比例系数
PIXEL2DIST_B = -0.3717         # 截距
PIXEL2DIST_SIMPLE_K = 100.16   # 简化公式系数
# 2. 距离→变焦倍数 拟合参数
DIST2ZOOM_A = 1.856            # 线性拟合斜率
DIST2ZOOM_B = -4.084           # 线性拟合截距

# ---------------------- 全局变量 ----------------------
_global_ptz_ser: Optional[serial.Serial] = None

# ---------------------- 内部工具函数 ----------------------
def _init_ptz_serial(is_query: bool = False) -> Optional[serial.Serial]:
    """初始化串口（复用全局对象）"""
    global _global_ptz_ser
    if _global_ptz_ser is not None and _global_ptz_ser.is_open:
        if _global_ptz_ser.timeout != (QUERY_TIMEOUT if is_query else CONTROL_TIMEOUT):
            _global_ptz_ser.timeout = QUERY_TIMEOUT if is_query else CONTROL_TIMEOUT
        return _global_ptz_ser

    try:
        with PTZ_LOCK:
            if _global_ptz_ser is None or not _global_ptz_ser.is_open:
                timeout = QUERY_TIMEOUT if is_query else CONTROL_TIMEOUT
                _global_ptz_ser = serial.Serial(
                    port=SERIAL_PORT,
                    baudrate=BAUD_RATE,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=timeout,
                    write_timeout=timeout
                )
        return _global_ptz_ser
    except serial.SerialException as e:
        print(f"❌ PTZ串口初始化失败: {e}")
        return None

def angle_to_ptz_bytes(target_angle: float) -> list[int]:
    """角度转云台指令4字节"""
    target_value = target_angle / ANGLE_COEFF
    if target_value < 0:
        target_value = 0x10000 + target_value
    target_value = int(round(target_value))
    p = (target_value >> 12) & 0x0F
    q = (target_value >> 8) & 0x0F
    r = (target_value >> 4) & 0x0F
    s = target_value & 0x0F
    return [0x00 | p, 0x00 | q, 0x00 | r, 0x00 | s]

def _zoom_multi_to_raw(zoom_multi: float) -> int:
    """
    内部函数：变焦倍数 → 云台raw值（线性映射）
    :param zoom_multi: 变焦倍数（1~25）
    :return: 对应的raw值（0~16384）
    """
    # 边界限制
    zoom_multi = max(ZOOM_RANGE[0], min(ZOOM_RANGE[1], zoom_multi))
    # 线性映射计算raw值
    zoom_raw = (zoom_multi - 1) * (ZOOM_RAW_MAX / (ZOOM_RANGE[1] - ZOOM_RANGE[0]))
    return int(round(zoom_raw))

def _zoom_raw_to_multi(zoom_raw: int) -> float:
    """
    内部函数：云台raw值 → 变焦倍数（反向映射）
    :param zoom_raw: 云台raw值（0~16384）
    :return: 对应的变焦倍数（1~25）
    """
    zoom_raw = max(0, min(ZOOM_RAW_MAX, zoom_raw))
    zoom_multi = (zoom_raw / ZOOM_RAW_MAX) * (ZOOM_RANGE[1] - ZOOM_RANGE[0]) + 1
    return round(zoom_multi, 1)

# ---------------------- 对外暴露：云台角度控制 ----------------------
def control_ptz_absolute(h_angle: float, v_angle: float, speed: int = DEFAULT_SPEED,
                         debug: bool = False) -> bool:
    """高频控制云台到绝对位置"""
    if not (ANGLE_RANGE_H[0] <= h_angle <= ANGLE_RANGE_H[1]) or \
       not (ANGLE_RANGE_V[0] <= v_angle <= ANGLE_RANGE_V[1]):
        if debug:
            print(f"❌ 角度超出范围：水平{h_angle}° | 垂直{v_angle}°")
        return False

    ser = _init_ptz_serial(is_query=False)
    if not ser:
        return False

    try:
        with PTZ_LOCK:
            h_bytes = angle_to_ptz_bytes(h_angle)
            v_bytes = angle_to_ptz_bytes(v_angle)
            cmd = [0x81, 0x01, 0x06, 0x02, speed, speed, *h_bytes, *v_bytes, 0xFF]
            ser.write(bytes(cmd))
            if debug:
                print(f"✅ 发送角度指令：水平{h_angle:.2f}° 垂直{v_angle:.2f}°")
        return True
    except Exception as e:
        if debug:
            print(f"❌ 角度控制失败: {e}")
        global _global_ptz_ser
        if _global_ptz_ser:
            try:
                _global_ptz_ser.close()
            except:
                pass
            _global_ptz_ser = None
        return False

# ---------------------- 对外暴露：云台变焦控制（核心修复） ----------------------
def control_ptz_zoom(zoom_multi: float, debug: bool = False) -> bool:
    """
    控制云台变焦（适配你的raw值参数）
    :param zoom_multi: 目标变焦倍数（1.0~25.0）
    :param debug: 是否打印调试信息
    :return: 执行成功返回True，失败返回False
    """
    # 1. 倍数→raw值转换
    zoom_raw = _zoom_multi_to_raw(zoom_multi)
    zoom_actual_multi = _zoom_raw_to_multi(zoom_raw)  # 实际生效的倍数
    if debug:
        print(f"📌 变焦映射：输入{zoom_multi}倍 → raw={zoom_raw} → 实际{zoom_actual_multi}倍")

    # 2. 拆分raw值为指令字节（关键！16位raw值拆分为2个8位字节）
    # 多数云台变焦指令需要将16位raw值拆分为高8位+低8位
    zoom_high = (zoom_raw >> 8) & 0xFF  # 高8位
    zoom_low = zoom_raw & 0xFF          # 低8位

    ser = _init_ptz_serial(is_query=False)
    if not ser:
        return False

    try:
        with PTZ_LOCK:
            # ---------------------- 关键：适配你的云台变焦指令格式 ----------------------
            # 通用VISCA变焦指令格式（适配16位raw值）：81 01 04 47 [高8位] [低8位] FF
            # 若你的云台指令格式不同，仅需修改此处cmd数组！
            zoom_cmd = [
                0x81, 0x01, 0x04, 0x47,  # 变焦指令前缀（通用VISCA协议）
                zoom_high, zoom_low,     # 16位raw值拆分的高低字节
                0xFF                     # 结束符
            ]

            # 发送指令
            ser.write(bytes(zoom_cmd))
            if debug:
                print(f"✅ 发送变焦指令：{[hex(b) for b in zoom_cmd]}")
                print(f"   raw值：{zoom_raw} → 高低字节：0x{zoom_high:02X} 0x{zoom_low:02X}")
        return True
    except Exception as e:
        if debug:
            print(f"❌ 变焦控制失败: {e}")
        global _global_ptz_ser
        if _global_ptz_ser:
            try:
                _global_ptz_ser.close()
            except:
                pass
            _global_ptz_ser = None
        return False

# ---------------------- 对外暴露：云台位置查询 ----------------------
def query_ptz_position(debug: bool = False) -> Optional[Tuple[float, float]]:
    """查询云台当前位置"""
    ser = _init_ptz_serial(is_query=True)
    if not ser:
        return None

    try:
        with PTZ_LOCK:
            ser.reset_input_buffer()
            ser.reset_output_buffer()

            query_cmd = b"\x81\x09\x06\x12\xFF"
            ser.write(query_cmd)
            time.sleep(0.2)

            feedback = ser.read(16)
            if debug:
                print(f"📤 查询反馈：{[hex(b) for b in feedback]}")

            if len(feedback) < 11:
                if debug:
                    print(f"❌ 反馈长度不足：{len(feedback)}字节")
                return None

            try:
                h_raw = (feedback[2] << 12) | (feedback[3] << 8) | (feedback[4] << 4) | feedback[5]
                h_signed = h_raw - 0x10000 if h_raw > 0x7FFF else h_raw
                h_angle = h_signed * ANGLE_COEFF

                v_raw = (feedback[6] << 12) | (feedback[7] << 8) | (feedback[8] << 4) | feedback[9]
                v_signed = v_raw - 0x10000 if v_raw > 0x7FFF else v_raw
                v_angle = v_signed * ANGLE_COEFF

                if not (-200 <= h_angle <= 200) or not (-50 <= v_angle <= 100):
                    if debug:
                        print(f"❌ 异常角度：水平{h_angle:.2f}° | 垂直{v_angle:.2f}°")
                    return None

                if debug:
                    print(f"🎯 当前位置：水平{h_angle:.2f}° | 垂直{v_angle:.2f}°")
                return (h_angle, v_angle)
            except IndexError:
                if debug:
                    print(f"❌ 解析失败：{[hex(b) for b in feedback]}")
                return None
    except Exception as e:
        if debug:
            print(f"❌ 查询失败: {e}")
        global _global_ptz_ser
        if _global_ptz_ser:
            try:
                _global_ptz_ser.close()
            except:
                pass
            _global_ptz_ser = None
        return None

# ---------------------- 对外暴露：足球像素→距离 ----------------------
def football_pixel2distance(d_pixel: float, use_simple: bool = False, debug: bool = False) -> Optional[float]:
    """足球像素直径 → 实际距离（m）"""
    if d_pixel <= 0:
        if debug:
            print(f"❌ 像素直径{d_pixel}无效（必须>0）")
        return None

    if use_simple:
        distance = PIXEL2DIST_SIMPLE_K / d_pixel
    else:
        distance = PIXEL2DIST_K / d_pixel + PIXEL2DIST_B

    distance = max(DIST_RANGE[0], min(DIST_RANGE[1], distance))
    if debug:
        print(f"📏 像素直径{d_pixel} → 距离{distance:.2f}m（简化公式：{use_simple}）")
    return round(distance, 2)

# ---------------------- 对外暴露：距离→变焦倍数（适配25倍） ----------------------
def distance2zoom(distance: float, debug: bool = False) -> Optional[float]:
    """实际距离 → 推荐云台变焦倍数（1~25倍）"""
    if distance < 0:
        if debug:
            print(f"❌ 距离{distance}m无效")
        return None

    # 线性拟合计算
    zoom = DIST2ZOOM_A * distance + DIST2ZOOM_B
    # 适配25倍最大变焦
    zoom = max(ZOOM_RANGE[0], min(ZOOM_RANGE[1], zoom))
    zoom = round(zoom, 1)

    if debug:
        print(f"🔍 距离{distance:.2f}m → 推荐变焦{zoom}倍（适配25倍上限）")
    return zoom

# ---------------------- 对外暴露：一键控制 ----------------------
def ptz_auto_control(h_angle: float, v_angle: float, football_pixel: float,
                     speed: int = DEFAULT_SPEED, debug: bool = False) -> dict:
    """一键控制：像素→距离→变焦→云台角度+变焦"""
    result = {
        "pixel2dist_success": False,
        "dist2zoom_success": False,
        "angle_control_success": False,
        "zoom_control_success": False,
        "distance": None,
        "zoom": None
    }

    # 像素→距离
    distance = football_pixel2distance(football_pixel, debug=debug)
    if distance:
        result["pixel2dist_success"] = True
        result["distance"] = distance

        # 距离→变焦
        zoom = distance2zoom(distance, debug=debug)
        if zoom:
            result["dist2zoom_success"] = True
            result["zoom"] = zoom

            # 控制变焦
            zoom_ok = control_ptz_zoom(zoom, debug=debug)
            result["zoom_control_success"] = zoom_ok

    # 控制角度
    angle_ok = control_ptz_absolute(h_angle, v_angle, speed, debug=debug)
    result["angle_control_success"] = angle_ok

    if debug:
        print(f"\n📊 一键控制结果：{result}")
    return result

# ---------------------- 对外暴露：关闭串口 ----------------------
def close_ptz_serial():
    """关闭串口（程序退出时必须调用）"""
    global _global_ptz_ser
    with PTZ_LOCK:
        if _global_ptz_ser is not None and _global_ptz_ser.is_open:
            _global_ptz_ser.close()
            _global_ptz_ser = None
            print("✅ PTZ串口已关闭")

# ---------------------- 测试代码（验证2.4倍变焦） ----------------------
if __name__ == "__main__":
    try:
        # 1. 测试2.4倍变焦（核心验证）
        print("=== 🔍 2.4倍变焦测试 ===")
        control_ptz_zoom(2.4, debug=True)  # 输入2.4倍，自动转raw值发送

        # 2. 测试像素→距离→变焦→控制全流程
        print("\n=== 🚀 全流程测试 ===")
        auto_result = ptz_auto_control(
            h_angle=0.0,
            v_angle=0.0,
            football_pixel=17,  # 实测像素直径
            debug=True
        )

        # 3. 查询最终位置
        print("\n=== 📍 最终位置查询 ===")
        query_ptz_position(debug=True)

    finally:
        close_ptz_serial()