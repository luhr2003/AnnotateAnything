from omegaconf import DictConfig
from magicsim.Env.Utils.file import Logger
from magicsim.Server.Utils import encode_data
import gymnasium as gym
import queue
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn
from loguru import logger as log
import argparse
import hydra


class GymService:
    def __init__(self, env_str, config: DictConfig, logger: Logger, port: int):
        self.config = config
        self.logger = logger
        self.env_str = env_str
        self.port = port
        self.create_env()

        # 请求和响应队列
        self.request_queue = queue.Queue()
        self.response_map = {}  # uuid -> response queue

        # 启动 FastAPI 服务器
        self.fastapi_app = FastAPI()
        self.setup_routes()
        self.fastapi_thread = None

    def create_env(self):
        self.env = gym.make(self.env_str, config=self.config, logger=self.logger)

    def setup_routes(self):
        @self.fastapi_app.post("/step")
        async def fastapi_step(data: dict):
            req_id = self._generate_request_id()
            self.request_queue.put(("step", data, req_id))
            return self._wait_for_response(req_id)

        @self.fastapi_app.post("/reset")
        async def fastapi_reset(data: dict):
            req_id = self._generate_request_id()
            self.request_queue.put(("reset", data, req_id))
            return self._wait_for_response(req_id)

        @self.fastapi_app.get("/render")
        async def fastapi_render():
            req_id = self._generate_request_id()
            self.request_queue.put(("render", {}, req_id))
            return self._wait_for_response(req_id)

        @self.fastapi_app.post("/close")
        async def fastapi_close():
            req_id = self._generate_request_id()
            self.request_queue.put(("close", {}, req_id))
            return self._wait_for_response(req_id)

    def _generate_request_id(self):
        import uuid

        cur_uuid = str(uuid.uuid4())
        while cur_uuid in self.response_map:
            cur_uuid = str(uuid.uuid4())
        return cur_uuid

    def _wait_for_response(self, req_id):
        resp_queue = queue.Queue()
        self.response_map[req_id] = resp_queue
        try:
            result = resp_queue.get()  # 可以设置超时防止卡死
            return Response(content=result)
        except queue.Empty:
            raise HTTPException(status_code=504, detail="Request Failed")

    def start_fastapi_server(self, host="0.0.0.0", port=8000):
        def run():
            uvicorn.run(self.fastapi_app, host=host, port=port)

        self.fastapi_thread = threading.Thread(target=run)
        self.fastapi_thread.start()

    def on_post_step(self):
        while not self.request_queue.empty():
            req_type, raw_data, req_id = self.request_queue.get()

            if req_id not in self.response_map:
                continue

            resp_queue = self.response_map.pop(req_id)

            try:
                if req_type == "step":
                    raw_data = raw_data.get("param", {})
                    result = self._handle_step(raw_data)
                elif req_type == "reset":
                    raw_data = raw_data.get("param", {})
                    result = self._handle_reset(raw_data)
                elif req_type == "render":
                    result = self._handle_render(raw_data)
                elif req_type == "close":
                    result = self._handle_close(raw_data)
                else:
                    result = {"error": "Unknown request type"}
            except Exception as e:
                self.logger.error(f"Error handling request {req_type}: {e}")
                self.logger.error(f"Raw data: {raw_data}")
                if req_type == "step":
                    result = encode_data((None, None, None, None, None))
                elif req_type == "reset":
                    result = encode_data((None, None))
                elif req_type == "render":
                    result = encode_data(None)
                elif req_type == "close":
                    result = {"status": "closed"}

            resp_queue.put(result)

    def _handle_step(self, raw_data):
        request_dict = raw_data
        param = request_dict
        response_dict = self.env.step(**param)
        return encode_data(response_dict)

    def _handle_reset(self, raw_data):
        request_dict = raw_data
        param = request_dict
        response_dict = self.env.reset(**param)
        return encode_data(response_dict)

    def _handle_render(self, raw_data):
        response_dict = self.env.render()
        return encode_data(response_dict)

    def _handle_close(self, raw_data):
        self.env.close()
        return {"status": "closed"}

    def run(self):
        self.env.reset(seed=0)
        self.on_post_reset()

        self.on_pre_run()

        # 启动 FastAPI 服务器
        self.start_fastapi_server(port=self.port)

        while self.env.app.is_running():
            self.env.sim_step()
            self.on_post_step()

        self.close_server()

    def close_server(self):
        self.on_post_run()
        self.env.stop()
        self.env.close()

    def on_pre_run(self):
        self.env.sim_step()

    def on_post_run(self):
        self.env.sim_step()

    def on_post_reset(self):
        self.env.sim_step()


def main():
    parser = argparse.ArgumentParser(description="Launch GymService Environment")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./Conf",
        help="Path to the configuration file for the environment.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="reach",
        help="Name of the configuration file to use.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./env_log",
        help="Directory to save the log file.",
    )
    parser.add_argument(
        "--env_id",
        type=int,
        default=0,
        help="ID of the environment instance to launch.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the FastAPI server.",
    )
    known_args, unknown_args = parser.parse_known_args()

    config_name = f"{known_args.env_name}_config"
    with hydra.initialize(version_base=None, config_path=known_args.config_path):
        cfg = hydra.compose(config_name=config_name)
    log_file = f"{known_args.log_dir}/{known_args.env_id}.log"
    env_str = cfg.env
    logger_name = f"{env_str}_{known_args.env_id}"
    logger = Logger(logger_name, log, log_file=log_file)
    env = GymService(env_str=env_str, config=cfg, logger=logger, port=known_args.port)
    env.run()


if __name__ == "__main__":
    main()
