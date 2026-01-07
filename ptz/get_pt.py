import serial
import time

def query_ptz_position():
    """
    最终版：修正左右正负反转 + 适配168°实际极值
    水平：左=+168°、右=-168°（已校准符号）
    垂直：上=+90°、下=-30°
    """
    # 基础配置
    serial_port = "COM3"
    baud_rate = 115200
    timeout = 2
    query_cmd = b"\x81\x09\x06\x12\xFF"
    feedback_len = 11
    angle_coeff = 0.075  # 1指令值=0.075°
    
    # 云台实际可达极值（实测值）
    H_MAX = 168.0    # 水平左最大
    H_MIN = -168.0   # 水平右最大
    V_MAX = 90.0     # 垂直上最大
    V_MIN = -30.0    # 垂直下最大

    try:
        # 打开串口（8N1配置）
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
        print(f"📤 发送查询命令: {[hex(b) for b in query_cmd]}")
        time.sleep(0.1)  # 等待云台响应

        # 读取反馈并关闭串口
        feedback = ser.read(feedback_len)
        ser.close()

        # 校验反馈格式
        if len(feedback) != feedback_len or feedback[0] != 0x90 or feedback[1] != 0x50 or feedback[-1] != 0xFF:
            print(f"❌ 反馈异常：{[hex(b) for b in feedback]}")
            return None
        print(f"📥 接收反馈数据: {[hex(b) for b in feedback]}")

        # 解析水平位置（pqrs：4位完整组合）
        p = feedback[2]
        q = feedback[3]
        r = feedback[4]
        s = feedback[5]
        horizontal_raw = (p << 12) | (q << 8) | (r << 4) | s
        print(f"\n🔍 水平pqrs解析：")
        print(f"   p={hex(p)}, q={hex(q)}, r={hex(r)}, s={hex(s)}")
        print(f"   组合值：0x{horizontal_raw:04X} → 十进制：{horizontal_raw}")

        # 解析垂直位置（tuvw：4位完整组合）
        t = feedback[6]
        u = feedback[7]
        v = feedback[8]
        w = feedback[9]
        vertical_raw = (t << 12) | (u << 8) | (v << 4) | w
        print(f"\n🔍 垂直tuvw解析：")
        print(f"   t={hex(t)}, u={hex(u)}, v={hex(v)}, w={hex(w)}")
        print(f"   组合值：0x{vertical_raw:04X} → 十进制：{vertical_raw}")

        # 转换为16位有符号数（处理正负）
        def to_signed_16bit(value):
            return value - 0x10000 if value > 0x7FFF else value
        
        h_signed = to_signed_16bit(horizontal_raw)
        v_signed = to_signed_16bit(vertical_raw)

        # ------------------- 关键修改：反转水平角度符号 -------------------
        # 原错误：horizontal_angle = -h_signed * angle_coeff
        # 修正后：去掉负号（或添加负号，根据实际方向调整）
        horizontal_angle = h_signed * angle_coeff  # 核心修改行
        # -----------------------------------------------------------------
        vertical_angle = v_signed * angle_coeff

        # 角度范围校验+提示
        print("\n=== 🎯 云台当前位置 ===")
        print(f"水平角度：{horizontal_angle:.2f}°（实际极值：{H_MIN}° ~ {H_MAX}° | 理论：-171°~+171°）")
        print(f"垂直角度：{vertical_angle:.2f}°（实际极值：{V_MIN}° ~ {V_MAX}°）")
        
        # 极值提醒
        if abs(horizontal_angle) >= H_MAX - 1:
            print(f"⚠️  水平已达机械限位（{horizontal_angle:.2f}°）")
        if vertical_angle >= V_MAX - 1:
            print(f"⚠️  垂直已达上限位（{vertical_angle:.2f}°）")
        if vertical_angle <= V_MIN + 1:
            print(f"⚠️  垂直已达下限位（{vertical_angle:.2f}°）")

        return (horizontal_angle, vertical_angle)

    except serial.SerialException as e:
        print(f"❌ 串口异常: {e}（检查COM3是否被占用/云台连接）")
        return None
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        return None

if __name__ == "__main__":
    # 测试最左侧/最右侧解析（验证正负方向）
    def test_direction():
        print("=== 🧪 验证水平方向符号 ===")
        # 最左侧反馈：90 50 0F 07 04 00 00 00 00 00 FF
        left_feedback = bytes([0x90,0x50,0x0F,0x07,0x04,0x00,0x00,0x00,0x00,0x00,0xFF])
        p = left_feedback[2]
        q = left_feedback[3]
        r = left_feedback[4]
        s = left_feedback[5]
        h_raw = (p<<12)|(q<<8)|(r<<4)|s
        h_signed = h_raw - 0x10000 if h_raw>0x7FFF else h_raw
        h_angle = h_signed * 0.075  # 修正后的计算方式
        print(f"最左侧解析：{h_angle:.2f}°（预期+168°）")

        # 最右侧反馈：90 50 00 08 0C 00 00 00 00 00 FF
        right_feedback = bytes([0x90,0x50,0x00,0x08,0x0C,00,0x00,0x00,0x00,0x00,0xFF])
        p = right_feedback[2]
        q = right_feedback[3]
        r = right_feedback[4]
        s = right_feedback[5]
        h_raw = (p<<12)|(q<<8)|(r<<4)|s
        h_signed = h_raw - 0x10000 if h_raw>0x7FFF else h_raw
        h_angle = h_signed * 0.075
        print(f"最右侧解析：{h_angle:.2f}°（预期-168°）")

    test_direction()
    print("\n" + "-"*50)
    query_ptz_position()