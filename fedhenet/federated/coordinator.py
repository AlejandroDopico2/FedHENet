import torch

# Imports for MQTT communication
import json
from loguru import logger

import tenseal as ts

from ..transport import MQTTTransport
from ..metrics import MetricsRecorder
from typing import Any
from ..algorithms.factory import get_algorithm


class Coordinator:
    def __init__(
        self,
        num_classes,
        device,
        num_clients: int,
        broker: str = "localhost",
        port: int = 1883,
        encrypted: bool = False,
        ctx: ts.Context | None = None,
        **kwargs: Any,
    ):
        if encrypted and (ctx and ctx.has_secret_key()):
            raise ValueError(
                "You passed a context with a private key to the coordinator"
            )

        self.ctx = ctx
        self.device = device
        self.encrypted = encrypted
        self.num_clients = num_clients

        self.algorithm = get_algorithm(
            device=device,
            num_classes=num_classes,
            encrypted=encrypted,
            ctx=ctx,
            **kwargs,
        )

        if self.algorithm.name == "fedhenet":
            self.extractor, self.rolann = self.algorithm.init_model()
        else:
            self.model = self.algorithm.init_model()

        # MQTT transport (connect explicitly and start loop)
        self.mqtt = MQTTTransport(client_id="coordinator", broker=broker, port=port)
        self.mqtt.connect()
        self.reset_topics(close_mqtt=False)

        self.mqtt.subscribe("federated/client/+/update", self._on_client_update)
        self.mqtt.loop_start()

        self._pending = []

    def reset_topics(self, close_mqtt: bool = True) -> None:
        """
        Cleanup retained MQTT topics and close the MQTT connection.
        Called from the CLI on normal completion or interruption.
        """
        try:
            self.mqtt.clean_topic("federated/global_model")
            for i in range(self.num_clients):
                self.mqtt.clean_topic(f"federated/client/{i}/update")
        except Exception as e:
            logger.warning(f"[Coordinator] Error during MQTT cleanup: {e}")
        try:
            if close_mqtt:
                self.mqtt.close()
        except Exception as e:
            logger.warning(f"[Coordinator] Error during MQTT shutdown: {e}")

    # Function to receive results from clients
    def _on_client_update(self, client, userdata, msg):
        # Ignore retained-clear messages (empty payload)
        if not msg.payload:
            logger.warning("[Coordinator] Empty payload received from client")
            return
        raw = msg.payload
        MetricsRecorder.instance().add_received_bytes(len(raw))
        data = json.loads(raw)  # Deserialize received message

        client_update = self.algorithm.deserialize_update(data, global_update=False)
        num_samples = client_update.get("metadata", {}).get("num_samples", 1)
        self._pending.append((num_samples, client_update))

        if len(self._pending) == self.num_clients:
            self._aggregate_and_broadcast()
            self._pending.clear()

    def _aggregate_and_broadcast(self):
        """Aggregate client updates using FedAvg and broadcast global model."""

        global_update = self.algorithm.aggregate_updates(
            self._pending,
        )

        if not self.encrypted and self.algorithm.name != "fedhenet":
            self.model = self.algorithm.apply_global(self.model, global_update)

        serialized_global = self.algorithm.serialize_global(
            global_update,
        )

        logger.info("Global update done! Publishing to clients")

        # Broadcast to all clients
        encoded = json.dumps(serialized_global)
        MetricsRecorder.instance().add_published_bytes(len(encoded.encode("utf-8")))
        self.mqtt.publish("federated/global_model", encoded, qos=1, retain=True)

    def evaluate(self, loader):
        correct = 0
        total = 0

        if self.algorithm.name == "fedhenet":
            self.extractor.to(self.device)
            self.rolann.to(self.device)
        else:
            self.model.to(self.device)

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                if self.algorithm.name == "fedhenet":
                    preds = self.rolann(self.extractor(x))
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
