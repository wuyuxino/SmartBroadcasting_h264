import serial
import time

# ---------------------- 共用配置 ----------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
TIMEOUT = 2
ANGLE_COEFF = 0.075  # 角度↔指令值的转换系数（1指令值=0.075°）
DEFAULT_SPEED = 0x08  # 默认速度（01~18h可选，08为中等速度）


def init_serial():
    """初始化串口（共用函数）"""
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT
        )
        return ser
    except serial.SerialException as e:
        print(f"❌ 串口初始化失败: {e}")
        return None


def angle_to_ptz_bytes(target_angle):
    """
    将目标角度转换为云台指令的4字节（0p 0q 0r 0s 或 0t 0u 0v 0w）
    返回：[0p, 0q, 0r, 0s]（4个字节）
    """
    # 1. 角度→指令值（带符号）
    target_value = target_angle / ANGLE_COEFF
    # 2. 转换为16位无符号补码（适配云台指令格式）
    if target_value < 0:
        target_value = 0x10000 + target_value  # 负数转16位无符号补码
    target_value = int(round(target_value))  # 取整（云台指令值为整数）
    # 3. 拆分为4个4位（p=高4位, q=次高4位, r=次低4位, s=低4位）
    p = (target_value >> 12) & 0x0F  # 高4位
    q = (target_value >> 8) & 0x0F   # 次高4位
    r = (target_value >> 4) & 0x0F   # 次低4位
    s = target_value & 0x0F          # 低4位
    # 4. 组合为指令要求的“0p 0q 0r 0s”（高4位固定为0）
    return [0x00 | p, 0x00 | q, 0x00 | r, 0x00 | s]


def send_absolute_position(h_angle, v_angle, speed=DEFAULT_SPEED):
    """
    发送绝对位置命令，控制云台移动到指定水平/垂直角度
    参数：h_angle（水平目标角度）、v_angle（垂直目标角度）、speed（速度）
    """
    ser = init_serial()
    if not ser:
        return False

    try:
        # 1. 角度→云台指令字节
        h_bytes = angle_to_ptz_bytes(h_angle)  # 水平：0p 0q 0r 0s
        v_bytes = angle_to_ptz_bytes(v_angle)  # 垂直：0t 0u 0v 0w
        # 2. 构造绝对位置指令
        # 格式：81 01 06 02 vv ww 0p 0q 0r 0s 0t 0u 0v 0w FF
        cmd = [
            0x81, 0x01, 0x06, 0x02,
            speed, speed,  # vv=水平速度, ww=垂直速度（统一用speed）
            *h_bytes,  # 水平位置：0p 0q 0r 0s
            *v_bytes,  # 垂直位置：0t 0u 0v 0w
            0xFF
        ]
        cmd_bytes = bytes(cmd)
        # 3. 发送指令
        ser.write(cmd_bytes)
        print(f"✅ 发送绝对位置指令：{[hex(b) for b in cmd_bytes]}")
        print(f"🎯 目标位置：水平{h_angle:.2f}° | 垂直{v_angle:.2f}°（速度：{speed}）")
        time.sleep(0.1)
        # 4. 读取云台响应（可选，部分云台会返回确认码）
        response = ser.read(11)
        if response:
            print(f"📥 云台响应：{[hex(b) for b in response]}")
        return True
    except Exception as e:
        print(f"❌ 发送指令失败: {e}")
        return False
    finally:
        if ser.is_open:
            ser.close()


def query_ptz_position():
    """原查询函数（已适配符号和实际极值）"""
    ser = init_serial()
    if not ser:
        return None

    try:
        # 发送查询命令
        query_cmd = b"\x81\x09\x06\x12\xFF"
        ser.write(query_cmd)
        print(f"📤 发送查询命令: {[hex(b) for b in query_cmd]}")
        time.sleep(0.1)
        # 读取反馈
        feedback = ser.read(11)
        if len(feedback) != 11 or feedback[0] != 0x90 or feedback[1] != 0x50 or feedback[-1] != 0xFF:
            print(f"❌ 反馈异常：{[hex(b) for b in feedback]}")
            return None
        # 解析水平/垂直位置
        # 水平：pqrs = feedback[2]-[5]
        h_raw = (feedback[2] << 12) | (feedback[3] << 8) | (feedback[4] << 4) | feedback[5]
        h_signed = h_raw - 0x10000 if h_raw > 0x7FFF else h_raw
        h_angle = h_signed * ANGLE_COEFF
        # 垂直：tuvw = feedback[6]-[9]
        v_raw = (feedback[6] << 12) | (feedback[7] << 8) | (feedback[8] << 4) | feedback[9]
        v_signed = v_raw - 0x10000 if v_raw > 0x7FFF else v_raw
        v_angle = v_signed * ANGLE_COEFF
        # 输出结果
        print("\n=== 🎯 当前云台位置 ===")
        print(f"水平角度：{h_angle:.2f}° | 垂直角度：{v_angle:.2f}°")
        return (h_angle, v_angle)
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return None
    finally:
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    # 步骤1：先查询当前位置（可选）
    print("=== 🔍 查询当前位置 ===")
    current_pos = query_ptz_position()
    print("-" * 50)

    # 步骤2：输入目标角度并控制
    try:
        target_h = float(input("请输入目标水平角度（范围：-168°~+168°）："))
        target_v = float(input("请输入目标垂直角度（范围：-30°~+90°）："))
        # 范围校验
        if not (-168 <= target_h <= 168):
            print("❌ 水平角度超出云台实际范围（-168°~+168°）")
        elif not (-30 <= target_v <= 90):
            print("❌ 垂直角度超出云台实际范围（-30°~+90°）")
        else:
            # 发送绝对位置命令
            print("\n=== 🚀 控制云台移动 ===")
            send_absolute_position(target_h, target_v)
            # 移动后再次查询（验证结果）
            print("\n=== 🔍 移动后位置 ===")
            query_ptz_position()
    except ValueError:
        print("❌ 输入无效，请输入数字！")