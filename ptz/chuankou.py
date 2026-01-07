import serial.tools.list_ports

def find_available_ports():
    ports = serial.tools.list_ports.comports()
    available_ports = []
    for port in ports:
        available_ports.append(port.device)
    return available_ports

if __name__ == "__main__":
    available_ports = find_available_ports()
    if available_ports:
        print("当前可用的串口有:")
        for port in available_ports:
            print(port)
    else:
        print("未找到可用的串口。")