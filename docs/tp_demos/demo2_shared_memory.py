"""
Demo 2: NanoVLLM TP 并行中的共享内存控制通道
=============================================

NanoVLLM 的多进程架构特点：
  - 使用 torch.multiprocessing "spawn" 模式，每个 rank 是一个独立进程
  - NCCL 负责张量数据的通信（GPU↔GPU）
  - SharedMemory + multiprocessing.Event 负责"控制面"通信：
    rank 0 告诉其他 rank 该调用哪个方法、传什么参数

设计原因：
  - spawn 模式下父子进程不共享 Python 对象
  - GPU 张量可通过 NCCL 同步，但 Python 层的调用指令（方法名、标量参数）
    需要轻量级的 IPC 机制 → 选用 POSIX SharedMemory + Event

关键代码位置：
  nanovllm/engine/model_runner.py
    - write_shm()  : rank 0 把 [method_name, *args] pickle 序列化写入 shm
    - read_shm()   : rank 1..N 等待 event，然后从 shm 反序列化读取
    - loop()       : rank 1..N 的控制循环
    - call()       : rank 0 的统一调用入口

运行方式（单机，不需要 GPU）：
  python demo2_shared_memory.py
"""

import pickle
import time
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: 最小化复现 NanoVLLM 的 SharedMemory + Event IPC 控制通道
# ─────────────────────────────────────────────────────────────────────────────

SHM_NAME = "nanovllm_demo"
SHM_SIZE = 2 ** 20   # 1 MB，与 NanoVLLM 源码保持一致


def write_shm(shm: SharedMemory, events: list, method_name: str, *args):
    """
    rank 0 调用：把 [method_name, *args] 序列化写入共享内存，再 set 所有 event。
    对应 model_runner.py:  ModelRunner.write_shm()
    """
    data = pickle.dumps([method_name, *args])
    n = len(data)
    # 前 4 字节存 payload 长度（小端）
    shm.buf[0:4] = n.to_bytes(4, "little")
    shm.buf[4:n + 4] = data
    print(f"[rank 0] write_shm: method={method_name!r}, args={args}, payload={n} bytes")
    for event in events:
        event.set()   # 通知每个 worker rank


def read_shm(shm: SharedMemory, event) -> tuple[str, list]:
    """
    rank 1..N 调用：等待 event，再从共享内存反序列化读取。
    对应 model_runner.py:  ModelRunner.read_shm()
    """
    event.wait()                           # 阻塞直到 rank 0 set()
    n = int.from_bytes(shm.buf[0:4], "little")
    method_name, *args = pickle.loads(shm.buf[4:n + 4])
    event.clear()                          # 清除 event，等待下次 set
    return method_name, args


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: 模拟 Worker 进程（rank 1..N 的 loop）
# ─────────────────────────────────────────────────────────────────────────────

class FakeWorker:
    """模拟 ModelRunner 中分布在其他 rank 的计算单元。"""

    def run(self, seqs: list, is_prefill: bool):
        print(f"[worker] run() called: seqs={seqs}, is_prefill={is_prefill}")

    def exit(self):
        print(f"[worker] exit() called, worker loop ending")


def worker_process(rank: int, event):
    """
    对应 model_runner.py:
        else:
            dist.barrier()
            self.shm = SharedMemory(name="nanovllm")
            self.loop()
    """
    # 打开 rank 0 已经创建好的 shm（rank 0 通过 barrier 确保先创建）
    shm = SharedMemory(name=SHM_NAME)
    worker = FakeWorker()

    print(f"[rank {rank}] shm opened, entering loop")

    # ── loop：不断从 shm 读取指令并执行 ──
    while True:
        method_name, args = read_shm(shm, event)
        print(f"[rank {rank}] loop: dispatching {method_name!r}  args={args}")
        method = getattr(worker, method_name, None)
        if method:
            method(*args)
        if method_name == "exit":
            break

    shm.close()
    print(f"[rank {rank}] worker_process done")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: 主进程（rank 0）的控制逻辑
# ─────────────────────────────────────────────────────────────────────────────

def rank0_main(events: list, world_size: int):
    """
    对应 model_runner.py:
        if rank == 0:
            self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
            dist.barrier()
            ...
            def call(self, method_name, *args):
                if self.world_size > 1 and self.rank == 0:
                    self.write_shm(method_name, *args)
                method = getattr(self, method_name)
                return method(*args)
    """
    shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    print(f"[rank 0] SharedMemory created: name={SHM_NAME}, size={SHM_SIZE}")

    # 等待所有 worker 进程就绪（实际代码用 dist.barrier()，这里用 sleep 模拟）
    time.sleep(0.3)

    # ── 模拟三次 call()：两次 run + 一次 exit ──
    print(f"\n[rank 0] === call 1: run(prefill) ===")
    write_shm(shm, events, "run", ["seq_a", "seq_b"], True)
    time.sleep(0.1)   # 实际中 rank 0 自己也在同步执行 run()

    print(f"\n[rank 0] === call 2: run(decode) ===")
    write_shm(shm, events, "run", ["seq_a", "seq_b"], False)
    time.sleep(0.1)

    print(f"\n[rank 0] === call 3: exit ===")
    write_shm(shm, events, "exit")
    time.sleep(0.2)

    shm.close()
    shm.unlink()   # 只有创建者负责 unlink，对应 model_runner.py exit()
    print(f"\n[rank 0] SharedMemory unlinked, done")


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: pickle 序列化细节——展示 write/read_shm 的数据格式
# ─────────────────────────────────────────────────────────────────────────────

def demo_pickle_format():
    """
    展示 NanoVLLM 在 shm 中存放的二进制格式：
      [ 4 bytes: payload length ] [ N bytes: pickle.dumps([method_name, *args]) ]
    """
    print("\n" + "="*60)
    print(" Part 4: pickle 序列化格式分析")
    print("="*60)

    method_name = "run"
    args = [["seq_1", "seq_2"], True]   # seqs + is_prefill
    payload = pickle.dumps([method_name, *args])
    n = len(payload)

    buf = bytearray(n + 4)
    buf[0:4] = n.to_bytes(4, "little")
    buf[4:n + 4] = payload

    # 反序列化
    n_read = int.from_bytes(buf[0:4], "little")
    reconstructed = pickle.loads(buf[4:n_read + 4])

    print(f"  序列化对象    : {[method_name, *args]}")
    print(f"  pickle bytes  : {n} 字节")
    print(f"  shm 头 4 字节 : {list(buf[0:4])} (小端整数 = {n})")
    print(f"  反序列化结果  : {reconstructed}")
    assert reconstructed == [method_name, *args]
    print("  ✓ 序列化/反序列化一致")


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: 与 SharedMemory 相关的生命周期管理
# ─────────────────────────────────────────────────────────────────────────────

def demo_shm_lifecycle():
    """
    演示 SharedMemory 的正确生命周期管理。
    对应 model_runner.py 的 exit() 方法：
        self.shm.close()
        dist.barrier()
        if self.rank == 0:
            self.shm.unlink()
    """
    print("\n" + "="*60)
    print(" Part 5: SharedMemory 生命周期管理")
    print("="*60)

    SHM_TMP = "nanovllm_lifecycle_demo"

    # 创建
    shm_creator = SharedMemory(name=SHM_TMP, create=True, size=1024)
    shm_creator.buf[0:5] = b"hello"
    print(f"  creator: wrote 'hello' to shm[0:5]")

    # 其他进程打开（模拟）
    shm_reader = SharedMemory(name=SHM_TMP)
    data = bytes(shm_reader.buf[0:5])
    print(f"  reader : read from shm[0:5] = {data}")
    assert data == b"hello"

    # 关闭（每个打开者都要 close）
    shm_reader.close()
    print(f"  reader : closed")

    shm_creator.close()
    shm_creator.unlink()   # 只有创建者 unlink，否则内存泄漏
    print(f"  creator: closed + unlinked")

    print("  ✓ SharedMemory 生命周期验证通过")

    # 验证 unlink 后无法再打开
    try:
        SharedMemory(name=SHM_TMP)
        print("  ✗ 应该抛出异常！")
    except FileNotFoundError:
        print(f"  ✓ unlink 后确认无法重新打开（FileNotFoundError 符合预期）")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Part 4 & 5 不需要多进程
    demo_pickle_format()
    demo_shm_lifecycle()

    # Part 1-3: 多进程 IPC 演示
    print("\n" + "="*60)
    print(" Part 1-3: SharedMemory + Event IPC 控制通道（多进程）")
    print("="*60)

    WORLD_SIZE = 3   # rank 0 + rank 1 + rank 2

    ctx = mp.get_context("spawn")

    # 为每个 worker rank（1..N-1）创建一个 Event
    events = [ctx.Event() for _ in range(WORLD_SIZE - 1)]

    # 启动 worker 进程
    workers = []
    for i in range(1, WORLD_SIZE):
        p = ctx.Process(target=worker_process, args=(i, events[i - 1]))
        p.start()
        workers.append(p)

    # rank 0 主逻辑（在主进程中运行）
    rank0_main(events, WORLD_SIZE)

    # 等待所有 worker 退出
    for p in workers:
        p.join()

    print("\n✓ 所有进程已退出，Demo 完成")
