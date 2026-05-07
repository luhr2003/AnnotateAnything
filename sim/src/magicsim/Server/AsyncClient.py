import requests
from flask import Flask, request, jsonify
import threading
import argparse
from magicsim.Server.Utils import encode_data

UPSTREAM_INFO = {
    "orion.iems.northwestern.edu": (9000, 10),
    "192.168.100.108": (8000, 8),
    # "nebula.osl.northwestern.edu": (9000, 10),
    # add more here [server_url: (port, env_num)]
}


class AsyncClient:
    def __init__(
        self,
        upstream_config: dict[str, tuple[int, int]],
        time_out: int,
        host="0.0.0.0",
        port=5001,
    ):
        """
        This is an AsyncClient which can change non-blocking requests to blocking requests.
        Initialize the Flask proxy server.

        :param upstream_config: dict[url, (port, env_num)]
        :param host: Bind host address
        :param port: Bind port number
        """
        self.env_num: int
        self.global_env_idx_to_upstream: dict[int, tuple[str, int]]
        self.global_env_idx_to_upstream_server_idx: dict[int, int]
        self.global_env_idx_to_kill_url_port: dict[int, tuple[str, int]]
        self.host = host
        self.port = port
        self.time_out = time_out
        self.app = Flask(__name__)
        self.server_thread = None
        self._process_upstream_config(upstream_config)
        self.setup_routes()

    def _process_upstream_config(self, upstream_config: dict[str, tuple[int, int]]):
        self.global_env_idx_to_upstream = {}
        self.global_env_idx_to_upstream_server_idx = {}
        self.global_env_idx_to_kill_url_port = {}
        self.all_servers = list(upstream_config.keys())
        curr_env_idx = 0
        for url, (upstream_port, upstream_num_envs) in upstream_config.items():
            url = url.rstrip("/")
            for server_env_idx in range(upstream_num_envs):
                self.global_env_idx_to_upstream[curr_env_idx] = (
                    url,
                    upstream_port + server_env_idx,
                )
                self.global_env_idx_to_upstream_server_idx[curr_env_idx] = (
                    server_env_idx
                )
                self.global_env_idx_to_kill_url_port[curr_env_idx] = (
                    url,
                    upstream_port - 1,
                )
                curr_env_idx += 1
        self.env_num = curr_env_idx

    def setup_routes(self):
        @self.app.route("/step", methods=["POST"])
        def proxy_step():
            try:
                rep = request.get_json()
                env_id = rep.get("env_id", 0)
                param = rep.get("param", {})
                if env_id not in self.global_env_idx_to_upstream:
                    raise ValueError("Invalid environment ID")
                return self._forward("/step", env_id=env_id, param=param)
            except Exception as e:
                print(f"Error in /step route: {e}")
                print("Error rep", request.get_json())
                return encode_data((None, None, None, None, None))

        @self.app.route("/reset", methods=["POST"])
        def proxy_reset():
            try:
                rep = request.get_json()
                env_id = rep.get("env_id", 0)
                param = rep.get("param", {})
                if env_id >= self.env_num:
                    raise ValueError("Invalid environment ID")
                return self._forward("/reset", env_id=env_id, param=param)
            except Exception as e:
                print(f"Error in /reset route: {e}")
                print("Error rep", request.get_json())
                return encode_data((None, None))

        @self.app.route("/render", methods=["GET"])
        def proxy_render():
            try:
                rep = request.get_json()
                env_id = rep.get("env_id", 0)
                if env_id >= self.env_num:
                    raise ValueError("Invalid environment ID")
                return self._forward("/render", env_id=env_id)
            except Exception as e:
                print(f"Error in /render route: {e}")
                print("Error rep", request.get_json())
                return encode_data(None)

        @self.app.route("/close", methods=["POST"])
        def proxy_close():
            rep = request.get_json()
            env_id = rep.get("env_id", 0)
            if env_id >= self.env_num:
                raise ValueError("Invalid environment ID")
            return self._forward("/close", env_id=env_id)

    def _forward(self, path: str, env_id: int, param: dict = None):
        (upstream_url, upstream_port) = self.global_env_idx_to_upstream[env_id]
        url = f"http://{upstream_url}:{upstream_port}{path}"
        try:
            if request.method == "POST":
                print("Forwarding POST request to:", url)
                resp = requests.post(
                    url,
                    json={"param": param},
                    headers=request.headers,
                    timeout=self.time_out,
                )
            else:
                resp = requests.get(url, headers=request.headers)
            if resp.status_code == 200:
                return resp.content
            else:
                raise requests.exceptions.RequestException(
                    f"Received {resp.status_code} from upstream server"
                )
        except requests.exceptions.RequestException as e:
            print(f"Error forwarding request to {url}: {e}")
            self.kill_env(env_id)
            if path == "/reset":
                return encode_data((None, None))
            elif path == "/step":
                return encode_data((None, None, None, None, None))
            elif path == "/render":
                return encode_data(None)
            return jsonify({"error": str(e)}), 500

    def start_flask(self):
        """Initialize and start the Flask server in a separate thread"""

        def run():
            self.app.run(host=self.host, port=self.port, threaded=True)

        self.server_thread = threading.Thread(target=run)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(
            f"Flask proxy server started on {self.host}:{self.port}, forwarding to {self.all_servers}"
            f" with a total of {self.env_num} environments."
        )

    def run(self):
        """运行 Flask 服务器"""
        self.start_flask()
        while True:
            pass  # This is a blocking call to keep the main thread alive

    def kill_env(self, env_id):
        url, port_for_kill = self.global_env_idx_to_kill_url_port[env_id]
        server_env_idx = self.global_env_idx_to_upstream_server_idx[env_id]
        url = f"http://{url}:{port_for_kill}/kill/{server_env_idx}"
        print(f"Killing environment {env_id} at {url}(local env id: {server_env_idx})")
        try:
            resp = requests.post(url)
            if resp.status_code == 200:
                print(f"Success: {resp.json()}")
            else:
                print(f"Failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"Request error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AsyncClient for forwarding requests to an upstream server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the Flask server",
    )
    parser.add_argument(
        "--port", type=int, default=7000, help="Port number to bind the Flask server"
    )
    parser.add_argument(
        "--time_out", type=int, default=60, help="Timeout for requests in seconds"
    )
    parser.add_argument(
        "--upstream_config",
        type=str,
        default=None,
        help="Path to json file with upstream server configuration",
    )

    args = parser.parse_args()
    if args.upstream_config is not None and args.upstream_config != "None":
        import json

        print(f"Loading upstream configuration from {args.upstream_config}")
        with open(args.upstream_config, "r") as f:
            upstream_config = json.load(f)
    else:
        print("No upstream configuration provided, using default.")
        upstream_config = UPSTREAM_INFO

    client = AsyncClient(
        upstream_config=upstream_config,
        host=args.host,
        port=args.port,
        time_out=args.time_out,
    )
    client.run()


if __name__ == "__main__":
    main()
