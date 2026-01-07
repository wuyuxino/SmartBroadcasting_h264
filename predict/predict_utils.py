"""
预测接口交互 - 预测数据组装、结果解析
"""
import copy
import time
import httpx
import orjson as fast_json
from config import settings

# Reuse a single httpx client to keep TCP connections alive and avoid
# recreating clients per request (reduces latency and CPU overhead).
_httpx_client = httpx.Client(verify=False)

def assemble_predict_data(cache, use_kalman=False, conf_thresh=None):
    """
    将缓存数据组装成指定的预测请求格式
    :param cache: 10帧缓存
    :param use_kalman: 是否启用卡尔曼滤波
    :param conf_thresh: 置信度阈值
    :return: 符合指定格式的字典
    """
    if conf_thresh is None:
        conf_thresh = settings.CONF_THRESHOLD
    
    frame_data = []
    for item in cache:
        # 跳过无有效目标的缓存项
        if not item.get("target_info"):
            continue
        target = item["target_info"]
        # 组装单帧数据
        frame_item = {
            "frame_id": item["frame_id"],
            "x1": float(target["center_x"]),
            "y1": float(target["center_y"]),
            "w": float(target["width"]),
            "h": float(target["height"]),
            "conf": float(target["conf"])
        }
        frame_data.append(frame_item)
    
    # 组装最终请求格式
    predict_data = {
        "frame_data": frame_data,
        "use_kalman": use_kalman,
        "conf_thresh": conf_thresh
    }
    return predict_data

def call_predict_api(request_data):
    """调用预测接口（优化版：httpx+orjson，复用连接，高性能）。

    使用模块级复用的 `_httpx_client`，并直接发送 orjson bytes，
    同时打印耗时与更精确的异常信息。
    """
    start = time.time()
    try:
        # 高性能序列化为 bytes（orjson 返回 bytes），避免中间 str decode/encode
        request_bytes = fast_json.dumps(request_data)

        response = _httpx_client.post(
            url=settings.PREDICT_API_URL,
            content=request_bytes,
            headers={"Content-Type": "application/json"},
            timeout=settings.REQUEST_TIMEOUT
        )

        elapsed = (time.time() - start) * 1000
        print(f"✅ 调用预测接口返回：status={response.status_code}，耗时={elapsed:.2f} ms")
        return response

    except httpx.ReadTimeout as e:
        elapsed = (time.time() - start) * 1000
        print(f"❌ 请求 ReadTimeout ({elapsed:.2f} ms)：{e}")
        return None
    except httpx.ConnectTimeout as e:
        elapsed = (time.time() - start) * 1000
        print(f"❌ 请求 ConnectTimeout ({elapsed:.2f} ms)：{e}")
        return None
    except httpx.HTTPError as e:
        elapsed = (time.time() - start) * 1000
        print(f"❌ 请求失败 ({elapsed:.2f} ms)：{e}")
        return None

def get_third_future_frame(result):
    """
    从预测结果中获取 future_frames 的第三个对象
    适配新接口返回格式（predictions列表），仅支持api_result类型
    :param result: last_predict_result 字典（仅api_result/None）
    :return: 第三个 future_frame 对象 / None（无数据/类型不匹配/入参None）
    """
    try:
        # 步骤1：入参None直接返回
        if result is None:
            print("⚠️ 入参为None（缓存不足/数据未变化），无预测结果")
            return None
        
        # 步骤2：仅处理api_result类型
        result_type = result.get('type')
        if result_type != "api_result":
            print(f"⚠️ 输入类型不匹配（当前type={result_type}），仅支持api_result")
            return None
        
        # 步骤3：解析新接口返回格式（核心修改）
        # 1. 取接口返回的根数据（原逻辑多了一层data，新格式直接取）
        api_data = result.get('data', {})
        if not api_data:
            print("⚠️ 接口返回数据为空")
            return None
        
        # 2. 取predictions列表（替代原「预测结果」列表）
        predictions = api_data.get('predictions', [])
        if not predictions:
            print("⚠️ predictions列表为空")
            return None
        
        # 3. 检查列表长度是否≥3（确保有第三个帧）
        if len(predictions) < 3:
            print(f"⚠️ predictions 长度不足3（当前{len(predictions)}个）")
            return None
        
        # 4. 取第三个元素（索引2）
        third_frame = predictions[2]
        print(f"✅ 成功提取第三个future_frame：{third_frame}")
        return third_frame
    
    except KeyError as e:
        print(f"❌ 获取第三个future_frame失败：关键字段缺失 {e}")
        return None
    except Exception as e:
        print(f"❌ 获取第三个future_frame失败：{e}")
        return None