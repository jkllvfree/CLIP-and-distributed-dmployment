import time
import torch


def run_timed_inference(
    tag: str,
    logger,
    device: torch.device,
    infer_func,
    local_ip: str = "",
    server_ip: str = "",
    one_way_ms: float = 0.0,
    extra_info: str = "",
    sync_cuda: bool = True,
    use_no_grad: bool = False,
):
    """
    通用推理测速函数：

    1. 可选 CUDA 同步（避免异步导致计时不准）
    2. 前后计时（time.perf_counter）
    3. 可选 torch.no_grad()
    4. 统一日志输出
    5. 返回 infer_func() 的输出

    参数：
        tag         : 日志标签，如 "attention" / "mlp" / "cos_sim"
        logger      : 日志对象，如 app.logger / client_logger
        device      : torch.device
        infer_func  : 无参函数，在其中写具体前向推理逻辑
        local_ip    : 本机 IP（服务端日志用，客户端可省略）
        server_ip   : 服务器 IP（服务端日志用，客户端可省略）
        one_way_ms  : 单向网络延迟 ms（服务端日志用，客户端可省略）
        extra_info  : 附加日志信息字符串
        sync_cuda   : 是否在计时前后做 torch.cuda.synchronize()
        use_no_grad : 是否在 torch.no_grad() 下执行 infer_func
    """
    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    if use_no_grad:
        with torch.no_grad():
            output = infer_func()
    else:
        output = infer_func()

    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    infer_ms = (t_end - t_start) * 1000.0

    if local_ip:
        # 服务端完整日志格式
        if extra_info:
            logger.info(
                "[%s] local_ip=%s server_ip=%s infer_ms=%.3f one_way_ms=%.3f type=%s %s",
                tag, local_ip, server_ip, infer_ms, one_way_ms, device, extra_info,
            )
        else:
            logger.info(
                "[%s] local_ip=%s server_ip=%s infer_ms=%.3f one_way_ms=%.3f type=%s",
                tag, local_ip, server_ip, infer_ms, one_way_ms, device,
            )
    else:
        # 客户端简化日志格式
        logger.info(
            "[%s] infer_ms=%.3f type=%s",
            tag, infer_ms, device,
        )

    return output
