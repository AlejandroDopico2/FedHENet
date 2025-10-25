from ..transport import MQTTTransport
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Imports for MQTT communication
import json
from ..metrics import MetricsRecorder
import tenseal as ts
from loguru import logger
import threading
from typing import Any
from ..algorithms.factory import get_algorithm


class Client:
    def __init__(
        self,
        num_classes,
        dataset,
        device,
        client_id: int,
        broker: str = "localhost",
        port: int = 1883,
        encrypted: bool = False,
        ctx: ts.Context | None = None,
        extractor: nn.Module | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ):
        self.device = device

        if encrypted and (ctx is None or not ctx.has_secret_key()):
            raise ValueError("For the client, context must include a private key")

        self.client_id = client_id
        self.encrypted = encrypted
        self.ctx = ctx
        # Store config for lazy init
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed + client_id)  # Different seed per client
        self.loader = DataLoader(
            dataset, batch_size=128, shuffle=True, generator=generator
        )
        self.num_samples = len(dataset)

        self.algorithm = get_algorithm(
            device=device,
            num_classes=num_classes,
            encrypted=encrypted,
            ctx=ctx,
            **kwargs,
        )
        self.model = self.algorithm.init_model()

        self._global_model_ready = threading.Event()

        self.mqtt = MQTTTransport(
            client_id=f"client_{client_id}", broker=broker, port=port
        )
        # Defer connection until publish/wait windows
        self.publish_topic = f"federated/client/{self.client_id}/update"

    def training(self, round_idx: int = 0, epochs: int = 1):
        """
        Train the local model using the algorithm.
        """
        model = self.algorithm.local_train(self.model, self.loader, epochs=epochs)

        if self.algorithm.name == "fedhenet":
            self.extractor, self.rolann = model
        else:
            self.model = model

        self.send_update(round_idx=round_idx)

    def send_update(self, round_idx: int = 0):
        """
        Compute and send the local model update to the coordinator.
        """
        self._global_model_ready.clear()

        if self.algorithm.name == "fedhenet":
            model = self.rolann
        else:
            model = self.model

        update = self.algorithm.compute_update(model)
        update["metadata"].update(
            {
                "client_id": self.client_id,
                "num_samples": self.num_samples,
                "round_idx": round_idx,
            }
        )

        serialized_update = self.algorithm.serialize_update(update)

        encoded = json.dumps(serialized_update)
        msg_size_bytes = len(encoded.encode("utf-8"))
        msg_size_mb = msg_size_bytes / (1024 * 1024)
        logger.info(
            f"[Client {self.client_id}] Serialized update size: {msg_size_mb:.3f} MB"
        )
        MetricsRecorder.instance().add_published_bytes(len(encoded.encode("utf-8")))

        # Connect just-in-time: subscribe to global, publish, wait, then disconnect
        try:
            self.mqtt.connect()
            self.mqtt.publish(self.publish_topic, encoded)
            self.mqtt.loop_start()
        finally:
            self.mqtt.unsubscribe(self.publish_topic)
            self.mqtt.subscribe("federated/global_model", self._on_global_model)

    def _on_global_model(self, client, userdata, msg):
        # Ignore retained-clear messages (empty payload)
        if not msg.payload:
            return
        data = json.loads(msg.payload)

        global_update = self.algorithm.deserialize_update(data, global_update=True)

        if self.algorithm.name == "fedhenet":
            self.rolann = self.algorithm.apply_global(self.rolann, global_update)
        else:
            # Ensure we keep the possibly-updated model reference
            self.model = self.algorithm.apply_global(self.model, global_update)

        self._global_model_ready.set()

    def evaluate(self, loader):
        correct = 0
        total = 0

        if self.algorithm.name == "fedhenet":
            self.extractor.to(self.device)
            self.rolann.to(self.device)
        else:
            self.model.to(self.device)
            self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if self.algorithm.name == "fedhenet":
                    features = self.extractor(x)
                    preds = self.rolann(features)
                else:
                    preds = self.model(x)

                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        if self.algorithm.name == "fedhenet":
            self.extractor.to("cpu")
            self.rolann.to("cpu")
        else:
            self.model.to("cpu")

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return correct / max(1, total)

    # Synchronization helpers
    def wait_for_global_model(self, timeout_s: float | None = 30.0) -> bool:
        return self._global_model_ready.wait(timeout=timeout_s)

    def has_global_model(self) -> bool:
        return self._global_model_ready.is_set()

    def shutdown(self) -> None:
        if not hasattr(self, "mqtt") or self.mqtt is None:
            return

        try:
            self.mqtt.close()
        except Exception as e:
            logger.warning(f"[Client {self.client_id}] Error during MQTT close: {e}")
