import os
import subprocess
import time
import argparse
import threading
from fastapi import FastAPI
import uvicorn

import psutil


class EnvLauncher:
    def __init__(
        self,
        config_path: str,
        env_name: str,
        log_dir: str,
        cuda_num: int,
        base_port: int,
        env_num: int,
        retry_interval: int,
        start_interval: int,
    ):
        self.config_path = config_path
        self.env_name = env_name
        self.log_dir = log_dir
        self.cuda_num = cuda_num
        self.base_port = base_port
        self.retry_interval = retry_interval
        self.env_num = env_num
        self.start_interval = start_interval
        self.env_processes = {}  # env_id: pid
        self.lock = threading.Lock()
        self.last_kill_time = {}  # env_id: timestamp

    def kill_process_tree(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                print(f"[INFO] Killing child process: {child.pid}")
                child.kill()
            print(f"[INFO] Killing parent process: {pid}")
            parent.kill()
        except psutil.NoSuchProcess:
            print(f"[INFO] No such process with PID: {pid}, already dead?")
        except Exception as e:
            print(f"[ERROR] Failed to kill process tree for PID {pid}: {e}")

    def kill_process_by_port(self, port):
        print(f"[INFO] Looking for processes using port: {port}")
        for conn in psutil.net_connections():
            if conn.status == "LISTEN" and conn.laddr.port == port:
                pid = conn.pid
                if pid:
                    print(f"[INFO] Found process using port {port}: PID={pid}")
                    self.kill_process_tree(pid)

    def kill_process_by_port_shell(self, port):
        try:
            cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}' | xargs kill -9"
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def launch_env(self, env_id):
        cuda_id = env_id % self.cuda_num
        port = self.base_port + env_id
        command = f"export CUDA_VISIBLE_DEVICES={cuda_id} && export NVIDIA_VISIBLE_DEVICES={cuda_id} && python -m magicsim.Server.GymServer --config_path {self.config_path} --env_name {self.env_name} --log_dir {self.log_dir} --env_id {env_id} --port {port}"
        while True:
            print(f"[INFO] Launching node: {command}", flush=True)
            try:
                process = subprocess.Popen(command, shell=True)
                with self.lock:
                    self.env_processes[env_id] = process.pid
                print(
                    f"[INFO] Process started with PID: {process.pid} (env_id={env_id})",
                    flush=True,
                )
                return_code = process.wait()
                if return_code != 0:
                    print(
                        f"[ERROR] Process (env_id={env_id}) exited with code: {return_code}",
                        flush=True,
                    )
                    raise subprocess.CalledProcessError(return_code, command)
            except Exception as e:
                print(
                    f"[ERROR] Command failed for env_id={env_id}: {e}. Retrying...",
                    flush=True,
                )

            if process is not None:
                self.kill_process_tree(process.pid)

            self.kill_process_by_port(port)
            self.kill_process_by_port_shell(port)

            print(
                f"[INFO] Waiting {self.retry_interval} seconds before retry for env_id={env_id}...",
                flush=True,
            )
            time.sleep(self.retry_interval)

    def kill_env_by_id(self, env_id):
        now = time.time()
        with self.lock:
            last_time = self.last_kill_time.get(env_id, 0)
            if now - last_time < 60:
                print(
                    f"[INFO] Kill for env_id={env_id} ignored (within 60s)", flush=True
                )
                return False
            pid = self.env_processes.get(env_id)
            self.last_kill_time[env_id] = now
        if pid is not None:
            print(f"[INFO] Killing env_id={env_id}, PID={pid}", flush=True)
            try:
                os.system(f"kill -9 {pid}")
            except Exception as e:
                print(f"[ERROR] Failed to kill process: {e}", flush=True)
            return True
        else:
            print(f"[INFO] No running process found for env_id={env_id}", flush=True)
            return False

    def start_kill_api(self):
        app = FastAPI()

        @app.post("/kill/{env_id}")
        def kill_env(env_id: int):
            print(f"[INFO] Received request to kill env_id={env_id}", flush=True)
            if 0 <= env_id < self.env_num:
                result = self.kill_env_by_id(env_id)
                if result:
                    return {"status": "ok", "msg": f"env_id {env_id} killed"}
                else:
                    return {
                        "status": "ignored",
                        "msg": f"env_id {env_id} kill ignored (within 60s or not running)",
                    }

        port = self.base_port - 1
        print(f"[INFO] FastAPI kill server listening on port {port}", flush=True)
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

    def launch_all(self):
        # 启动 FastAPI 杀进程服务
        api_thread = threading.Thread(target=self.start_kill_api, daemon=True)
        api_thread.start()
        threads = []
        for env_id in range(self.env_num):
            t = threading.Thread(target=self.launch_env, args=(env_id,))
            t.daemon = True
            t.start()
            threads.append(t)
            time.sleep(self.start_interval)
        # 主线程等待所有子线程
        for t in threads:
            t.join()


def main():
    parser = argparse.ArgumentParser(description="Launch multiple IsaacSimEnv")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./Conf",
        help="Path to the configuration file for the environment.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Environment Name",
    )
    parser.add_argument(
        "--env_num", type=int, default=4, help="Number of envs to launch."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./env_log",
        help="Directory to save the log file.",
    )
    parser.add_argument(
        "--retry_interval", type=int, default=3, help="Retry interval in seconds."
    )
    parser.add_argument(
        "--cuda_num",
        type=int,
        default=4,
        help="CUDA device number to use for the environment.",
    )
    parser.add_argument(
        "--start_interval",
        type=int,
        default=40,
        help="Interval in seconds to wait before starting the next environment.",
    )
    parser.add_argument(
        "--base_port",
        type=int,
        required=True,
        help="Base port number for the environment.",
    )
    args = parser.parse_args()

    launcher = EnvLauncher(
        config_path=args.config_path,
        env_name=args.env_name,
        log_dir=args.log_dir,
        cuda_num=args.cuda_num,
        base_port=args.base_port,
        env_num=args.env_num,
        retry_interval=args.retry_interval,
        start_interval=args.start_interval,
    )
    launcher.launch_all()


if __name__ == "__main__":
    main()
