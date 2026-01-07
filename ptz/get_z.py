import serial
import time

def query_ptz_zoom():
    """
    云台变焦查询（基于焦距的非线性倍数计算）
    核心逻辑：
    1. raw值（0~16384）线性映射到焦距（7.1mm~171.95mm）
    2. 变焦倍数 = 当前焦距 ÷ 广角端焦距（7.1mm）
    """
    # 基础配置
    serial_port = "COM3"
    baud_rate = 9600
    timeout = 2
    query_cmd = b"\x81\x09\x04\x47\xFF"  # 变焦查询指令
    feedback_len = 7                     # 反馈长度7字节

    # 焦距&变焦参数（来自你的镜头参数）
    focal_wide = 7.1                     # 广角端焦距（对应1倍、raw=0）
    focal_tele = 171.95                  # 远端焦距（对应标称25倍、raw=16384）
    zoom_raw_max = 16384                 # raw最大值（对应远端焦距）

    try:
        # 打开串口
        ser = serial.Serial(
            port=serial_port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )
        print(f"✅ 成功打开串口 {serial_port}")

        # 发送查询命令
        ser.write(query_cmd)
        print(f"📤 发送变焦查询命令: {[hex(b) for b in query_cmd]}")
        time.sleep(0.1)

        # 读取反馈并关闭串口
        feedback = ser.read(feedback_len)
        ser.close()

        # 校验反馈格式
        if len(feedback) != feedback_len or feedback[0] != 0x90 or feedback[1] != 0x50 or feedback[-1] != 0xFF:
            print(f"❌ 变焦反馈异常：{[hex(b) for b in feedback]}")
            return None
        print(f"📥 接收变焦反馈数据: {[hex(b) for b in feedback]}")

        # 解析raw值
        p = feedback[2]
        q = feedback[3]
        r = feedback[4]
        s = feedback[5]
        zoom_raw = (p << 12) | (q << 8) | (r << 4) | s
        print(f"\n🔍 变焦pqrs解析：")
        print(f"   p={hex(p)}, q={hex(q)}, r={hex(r)}, s={hex(s)}")
        print(f"   组合值：0x{zoom_raw:04X} → 十进制：{zoom_raw}")

        # 1. 先将raw值线性映射到实际焦距
        # 公式：当前焦距 = 广角焦距 + (raw/raw最大值) × (远端焦距-广角焦距)
        current_focal = focal_wide + (zoom_raw / zoom_raw_max) * (focal_tele - focal_wide)
        
        # 2. 再计算真实变焦倍数（=当前焦距÷广角焦距）
        real_zoom = current_focal / focal_wide

        # 输出结果
        print("\n=== 🔍 云台当前变焦状态 ===")
        print(f"当前焦距：{current_focal:.2f}mm（范围：{focal_wide}mm~{focal_tele}mm）")
        print(f"真实变焦倍数：{real_zoom:.1f}倍（标称最大值：25倍）")
        
        # 极值提醒
        if real_zoom >= (focal_tele / focal_wide) - 0.2:
            print(f"⚠️  变焦已达远端（标称25倍）")
        if real_zoom <= 1.1:
            print(f"⚠️  变焦已达广角端（1倍）")

        return (current_focal, real_zoom)

    except serial.SerialException as e:
        print(f"❌ 串口异常: {e}（检查COM3是否被占用/云台连接）")
        return None
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        return None

def test_zoom_parsing():
    """测试非线性倍数解析（匹配焦距参数）"""
    print("=== 🧪 验证焦距&非线性倍数解析 ===")
    # 测试1：广角端（raw=0 → 7.1mm → 1倍）
    zoom_raw = 0
    current_focal = 7.1 + (zoom_raw/16384)*(171.95-7.1)
    real_zoom = current_focal /7.1
    print(f"raw=0 → 焦距={current_focal:.2f}mm → 倍数={real_zoom:.1f}倍（预期1.0倍）")

    # 测试2：raw=1000（模拟1倍→2倍之间的状态）
    zoom_raw = 1000
    current_focal = 7.1 + (zoom_raw/16384)*(171.95-7.1)
    real_zoom = current_focal /7.1
    print(f"raw=1000 → 焦距={current_focal:.2f}mm → 倍数={real_zoom:.1f}倍（符合“调10几次到2倍”）")

    # 测试3：远端（raw=16384 → 171.95mm → 约24.2倍，标称25倍）
    zoom_raw = 16384
    current_focal = 7.1 + (zoom_raw/16384)*(171.95-7.1)
    real_zoom = current_focal /7.1
    print(f"raw=16384 → 焦距={current_focal:.2f}mm → 倍数={real_zoom:.1f}倍（标称25倍）")

if __name__ == "__main__":
    test_zoom_parsing()
    print("\n" + "-"*50)
    query_ptz_zoom()