# utils/offloader.py
import torch
import io
import base64
import requests
import time
import logging


class OffloadHandler:
    def __init__(self, server_ip, server_port, config, logger=None):
        self.server_ip = server_ip
        self.server_port = server_port
        self.config = config
        self.logger = logger or logging.getLogger('client')

    def should_offload(self, module_type: str) -> bool:
        """
        判断当前模块是否需要卸载到服务器。

        参数:
            module_type: 模块标识，对应 OFFLOAD_CONFIG 中的 key，
                        如 'visual_attn' / 'text_mlp' / 'vision_conv' / 'complete_encoders' 等

        TODO: 后续可扩展为逐层粒度的卸载控制。
              方案思路：
              1. 增加 layer_id: int 和 encoder_type: str 参数
              2. 拼接 key = f"{encoder_type}_{module_type}_{layer_id}"
                 （如 visual_attn_5 表示只卸载视觉编码器第5层attention）
              3. 在 OFFLOAD_CONFIG 中增加 per-layer 环境变量
                 （如 OFFLOAD_VISUAL_ATTN_5）
              4. 查询时优先查细粒度 key，未命中则回落粗粒度 key
              当前设计为模块级粒度（如 visual_attn 整体卸载）。
        """
        return self.config.get(module_type, False)

    def call_remote(self, endpoint: str, data_dict: dict, device: torch.device, fallback_fn=None):
        """
        序列化数据 -> 发送请求 -> 反序列化结果
        如果远程调用失败且提供了 fallback_fn，则降级执行本地计算。
        """
        try:
            # 1. 序列化
            buffer = io.BytesIO()
            # 将 tensor 转为 cpu 以便序列化
            cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
            torch.save(cpu_data, buffer)
            data_str = base64.b64encode(buffer.getvalue()).decode()

            # 2. 发送请求
            url = f"http://{self.server_ip}:{self.server_port}/{endpoint}"
            payload = {
                "data": data_str,
                "client_send_ts": time.time()
            }

            t_start = time.perf_counter()
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            resp_json = resp.json()
            http_ms = (time.perf_counter() - t_start) * 1000

            if self.logger:
                self.logger.info(f"[{endpoint}] server={self.server_ip} rtt={http_ms:.2f}ms type=传输")

            # 3. 反序列化
            output_str = resp_json['output']
            output_buffer = io.BytesIO(base64.b64decode(output_str))
            output_dict = torch.load(output_buffer)

            # 移回原设备
            return output_dict['output'].to(device)

        except Exception as e:
            if fallback_fn is not None:
                if self.logger:
                    self.logger.warning(
                        "[%s] 远程调用失败，降级本地计算: %s", endpoint, e
                    )
                return fallback_fn()
            if self.logger:
                self.logger.error(f"[{endpoint}] Failed: {e}")
            raise