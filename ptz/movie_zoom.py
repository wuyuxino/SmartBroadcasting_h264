import serial
import time

# ---------------------- 变焦专用配置 ----------------------
SERIAL_PORT = "COM3"
BAUD_RATE = 9600
TIMEOUT = 2

# 焦距&变焦核心参数（适配你的云台）
FOCAL_WIDE = 7.1               # 广角端焦距（对应raw=0，1倍）
FOCAL_TELE_NOM = 7.1 * 25      # 标称远端焦距（177.5mm，对应raw=16384，25倍）
ZOOM_RAW_MAX = 16384            # 变焦raw值最大值（对应25倍）


def init_serial():
    """初始化串口"""
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


def zoom_value_to_bytes(target_raw):
    """
    将变焦raw值转换为云台指令的4字节（0p 0q 0r 0s）
    参数：target_raw - 变焦原始值（0~16384）
    返回：[0p, 0q, 0r, 0s]（4个字节）
    """
    # 确保raw值在有效范围
    target_raw = max(0, min(ZOOM_RAW_MAX, int(round(target_raw))))
    # 拆分为4个4位（p=高4位, q=次高4位, r=次低4位, s=低4位）
    p = (target_raw >> 12) & 0x0F
    q = (target_raw >> 8) & 0x0F
    r = (target_raw >> 4) & 0x0F
    s = target_raw & 0x0F
    # 组合为指令字节（高4位固定为0）
    return [0x00 | p, 0x00 | q, 0x00 | r, 0x00 | s]


def focal_to_zoom_raw(focal_length):
    """
    将焦距转换为变焦raw值（线性映射）
    参数：focal_length - 目标焦距（mm）
    返回：对应的raw值（0~16384）
    """
    # 焦距范围校验
    focal_length = max(FOCAL_WIDE, min(FOCAL_TELE_NOM, focal_length))
    # 线性映射公式
    zoom_raw = (focal_length - FOCAL_WIDE) / (FOCAL_TELE_NOM - FOCAL_WIDE) * ZOOM_RAW_MAX
    return zoom_raw


def zoom_multiple_to_raw(zoom_multiple):
    """
    将变焦倍数转换为raw值（适配标称倍数：1~25倍）
    参数：zoom_multiple - 目标变焦倍数
    返回：对应的raw值（0~16384）
    """
    # 倍数范围校验
    zoom_multiple = max(1.0, min(25.0, zoom_multiple))
    # 倍数→焦距→raw值
    focal_length = FOCAL_WIDE * zoom_multiple
    return focal_to_zoom_raw(focal_length)


def send_zoom_command(target_raw):
    """
    发送变焦控制命令：81 01 04 47 0p 0q 0r 0s FF
    参数：target_raw - 变焦原始值（0~16384）
    """
    ser = init_serial()
    if not ser:
        return False

    try:
        # 1. 转换raw值为指令字节（0p 0q 0r 0s）
        zoom_bytes = zoom_value_to_bytes(target_raw)
        # 2. 构造变焦控制指令
        cmd = [
            0x81, 0x01, 0x04, 0x47,
            *zoom_bytes,  # 0p 0q 0r 0s
            0xFF
        ]
        cmd_bytes = bytes(cmd)
        # 3. 发送指令
        ser.write(cmd_bytes)
        print(f"✅ 发送变焦控制指令：{[hex(b) for b in cmd_bytes]}")
        # 计算并显示对应参数
        calc_focal = FOCAL_WIDE + (target_raw/ZOOM_RAW_MAX)*(FOCAL_TELE_NOM - FOCAL_WIDE)
        calc_multiple = calc_focal / FOCAL_WIDE
        print(f"🎯 目标参数：")
        print(f"   变焦raw值：{target_raw}")
        print(f"   对应焦距：{calc_focal:.2f}mm")
        print(f"   对应倍数：{calc_multiple:.1f}倍")
        time.sleep(0.2)  # 给云台响应时间
        # 4. 读取云台响应（可选）
        response = ser.read(7)  # 变焦响应固定7字节
        if response:
            print(f"📥 云台变焦响应：{[hex(b) for b in response]}")
        return True
    except Exception as e:
        print(f"❌ 发送变焦指令失败: {e}")
        return False
    finally:
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    print("=== 🔍 云台变焦控制 ===")
    # 选择输入类型
    zoom_type = input("请选择输入类型（1=焦距(mm) | 2=变焦倍数）：")
    try:
        if zoom_type == "1":
            # 输入焦距
            focal = float(input(f"请输入目标焦距（范围：{FOCAL_WIDE}~{FOCAL_TELE_NOM:.1f}mm）："))
            print(f"\n📌 正在变焦到 {focal:.2f}mm...")
            zoom_raw = focal_to_zoom_raw(focal)
            send_zoom_command(zoom_raw)
        elif zoom_type == "2":
            # 输入倍数
            multiple = float(input("请输入目标变焦倍数（范围：1.0~25.0倍）："))
            print(f"\n📌 正在变焦到 {multiple:.1f}倍...")
            zoom_raw = zoom_multiple_to_raw(multiple)
            send_zoom_command(zoom_raw)
        else:
            print("❌ 无效选择，请输入1或2！")
    except ValueError:
        print("❌ 输入无效，请输入数字！")