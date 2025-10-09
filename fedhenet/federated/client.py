import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np

# Imports for MQTT communication
import json
import pickle
import base64
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
from ..rolann import ROLANN
from ..metrics import MetricsRecorder
import tenseal as ts
from loguru import logger
import threading


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
    ):
        self.device = device

        if encrypted and (ctx is None or not ctx.has_secret_key()):
            raise ValueError("For the client, context must include a private key")

        self.encrypted = encrypted
        self.rolann = ROLANN(num_classes=num_classes, encrypted=encrypted, context=ctx)

        # Use a generator for reproducible shuffling
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed + client_id)  # Different seed per client
        self.loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)  # Local dataset

        # Feature extractor (default: pretrained frozen ResNet18)
        if extractor is None:
            extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
            extractor.fc = nn.Identity()
            for p in extractor.parameters():
                p.requires_grad = False
        self.extractor = extractor
        # Keep models on CPU by default; move to target device only when needed
        self.extractor.eval()
        self.extractor.to("cpu")
        self.rolann.to("cpu")

        # MQTT configuration
        self.mqtt = mqtt.Client(
            client_id=f"client_{client_id}",
            callback_api_version=CallbackAPIVersion.VERSION1,
        )  # mqtt for each client
        self.mqtt.message_callback_add(
            "federated/global_model", self._on_global_model
        )  # callback to receive the global model
        self.mqtt.connect(broker, port)  # Connect to the MQTT broker
        # Subscribe with qos=1 to get retained last message if coordinator already published
        self.mqtt.subscribe(
            "federated/global_model", qos=1
        )  # Subscribe to the global model topic
        self.mqtt.loop_start()  # Start the message loop

        self.client_id = client_id  # Client ID
        self._global_model_ready = threading.Event()

    def _process_label(self, y):
        return (
            torch.nn.functional.one_hot(y, num_classes=self.rolann.num_classes) * 0.9
            + 0.05
        )

    def training(self):
        """
        Iterate over the local dataset, extract features using the local ResNet and
        update the ROLANN layer
        """
        self.extractor.to(self.device)
        self.rolann.to(self.device)

        for x, y in self.loader:
            x = x.to(self.device)
            y = self._process_label(y).to(self.device)

            with torch.no_grad():
                features = self.extractor(x)

            self.rolann.aggregate_update(features, y)

        # Move models back to CPU to free VRAM
        self.extractor.to("cpu")
        self.rolann.to("cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.aggregate_parcial()

    def aggregate_parcial(self):
        """
        Publish the local model to the MQTT broker
        """
        # Return the accumulated matrices M and US for each class
        local_M = self.rolann.mg
        local_US = [
            torch.matmul(
                self.rolann.ug[i], torch.diag(self.rolann.sg[i].clone().detach())
            )
            for i in range(self.rolann.num_classes)
        ]

        # Serialize and publish the update with metadata envelope
        payload = []
        for M_enc, US in zip(local_M, local_US):
            if self.encrypted:
                serialized = M_enc.serialize()
                bM = base64.b64encode(serialized).decode()
                m_format = "ckks"
            else:
                m_plain = M_enc.cpu().numpy().tolist()
                bM = base64.b64encode(pickle.dumps(m_plain)).decode()
                m_format = "tensor"

            bUS = base64.b64encode(pickle.dumps(US.cpu().numpy())).decode()
            payload.append({"M": bM, "US": bUS})

        envelope = {
            "version": 1,
            "format": m_format,  # format of M entries
            "client_id": self.client_id,
            "num_classes": self.rolann.num_classes,
            "payload": payload,
        }

        topic = f"federated/client/{self.client_id}/update"
        encoded = json.dumps(envelope)
        MetricsRecorder.instance().add_published_bytes(len(encoded.encode("utf-8")))
        self.mqtt.publish(topic, encoded, qos=1)
        # Optionally free memory hints
        try:
            del payload, envelope, encoded
        except Exception:
            pass

    # Receives the global model and decomposes it into M and US matrices
    def _on_global_model(self, client, userdata, msg):
        # Ignore retained-clear messages (empty payload)
        if not msg.payload:
            return
        data = json.loads(msg.payload)
        # Accept both envelope with payload and raw list for backward compatibility
        items = (
            data.get("payload")
            if isinstance(data, dict) and "payload" in data
            else data
        )
        mg, ug, sg = [], [], []
        for i in items:
            m_bytes = base64.b64decode(i["M"])

            # if CKKS ciphertext, reconstruct, otherwise pickle
            try:
                M_enc = ts.ckks_vector_from(self.rolann.context, m_bytes)
                mg.append(M_enc)
            except Exception:
                arr = pickle.loads(m_bytes)
                mg.append(
                    torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)
                )

            US_np = pickle.loads(base64.b64decode(i["US"]))

            # Decompose US into U and S
            U, S, _ = torch.linalg.svd(
                torch.from_numpy(US_np).to(self.device), full_matrices=False
            )
            ug.append(U)
            sg.append(S)

        # Update the accumulated matrices of ROLANN
        self.rolann.mg = mg
        self.rolann.ug = ug
        self.rolann.sg = sg
        self.rolann._calculate_weights()
        self._global_model_ready.set()

    def evaluate(self, loader):
        correct = 0
        total = 0

        self.extractor.to(self.device)
        self.rolann.to(self.device)

        # Guard: no weights computed yet
        w_list = getattr(self.rolann, "w", None)
        if not isinstance(w_list, list) or len(w_list) == 0:
            logger.warning(
                f"Client {self.client_id}: no weights available; skipping evaluation"
            )
            return 0.0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                features = self.extractor(x)
                preds = self.rolann(features)
                if preds.shape[-1] == 0:
                    logger.error(
                        f"Client {self.client_id}: preds has zero classes. w_len={len(w_list)}; num_classes={getattr(self.rolann, 'num_classes', None)}"
                    )
                    # Free memory before returning
                    self.extractor.to("cpu")
                    self.rolann.to("cpu")
                    if self.device == "cuda" and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return 0.0

                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        self.extractor.to("cpu")
        self.rolann.to("cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return correct / max(1, total)

    # Synchronization helpers
    def wait_for_global_model(self, timeout_s: float | None = 30.0) -> bool:
        return self._global_model_ready.wait(timeout=timeout_s)

    def has_global_model(self) -> bool:
        return self._global_model_ready.is_set()

    def shutdown(self) -> None:
        try:
            self.mqtt.loop_stop()
        except Exception:
            pass
        try:
            self.mqtt.disconnect()
        except Exception:
            pass
