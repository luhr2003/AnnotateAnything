import subprocess
import time
import argparse

import psutil


class AsyncRestAPILauncher:
    def __init__(self, upstream_config, host, port, time_out, retry_interval=5):
        self.upstream_config = upstream_config
        self.host = host
        self.port = port
        self.time_out = time_out

        self.retry_interval = retry_interval

    def kill_process_by_port(self, port):
        print(f"[INFO] Looking for processes using port: {port}")
        for conn in psutil.net_connections():
            if conn.status == "LISTEN" and conn.laddr.port == port:
                pid = conn.pid
                if pid:
                    print(f"[INFO] Found process using port {port}: PID={pid}")
                    self.kill_process_tree(pid)

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

    def kill_process_by_port_shell(self, port):
        try:
            cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}' | xargs kill -9"
            subprocess.run(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass

    def launch(self):
        while True:
            # Construct command with namespace
            command = f"python -m magicsim.Server.AsyncClient --upstream_config {self.upstream_config} --time_out {self.time_out} --host {self.host} --port {self.port}"
            print(f"[INFO] Launching rest: {command}")

            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                )
                print(f"[INFO] Process started with PID: {process.pid}")

                return_code = process.wait()
                if return_code != 0:
                    print(
                        f"[ERROR] Process exited unexpectedly with code: {return_code}"
                    )
                    raise subprocess.CalledProcessError(return_code, command)

            except Exception as e:
                print(f"Command failed with return code {e.returncode}. Retrying...")

            if process is not None:
                self.kill_process_tree(process.pid)

            self.kill_process_by_port(self.port)

            self.kill_process_by_port_shell(self.port)

            print(f"[INFO] Waiting {self.retry_interval} seconds before retry...")
            time.sleep(self.retry_interval)


def main():
    parser = argparse.ArgumentParser(description="Launch RestAPI")
    parser.add_argument(
        "--upstream_config",
        type=str,
        default=None,
        help="Path to json file with upstream server configuration",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the REST API server.",
    )
    parser.add_argument(
        "--port", type=int, required=True, help="Port number for the REST API."
    )
    parser.add_argument(
        "--time_out", type=int, required=True, help="Timeout for requests in seconds."
    )
    parser.add_argument(
        "--retry_interval",
        type=int,
        default=5,
        help="Interval in seconds to wait before retrying the launch if it fails.",
    )
    args = parser.parse_args()

    launcher = AsyncRestAPILauncher(
        upstream_config=args.upstream_config,
        host=args.host,
        port=args.port,
        time_out=args.time_out,
        retry_interval=args.retry_interval,
    )

    launcher.launch()


if __name__ == "__main__":
    main()
