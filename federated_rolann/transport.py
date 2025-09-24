from typing import Callable

from paho.mqtt.client import Client as MQTTClient
from paho.mqtt.client import CallbackAPIVersion


class MQTTTransport:
    """
    Minimal transport wrapper around paho-mqtt used by coordinator/client.
    Keeps the same callback signature as paho for easy drop-in.
    """

    def __init__(self, client_id: str, broker: str, port: int) -> None:
        self.client = MQTTClient(client_id=client_id, callback_api_version=CallbackAPIVersion.VERSION1)
        self.client.connect(broker, port)

    def subscribe(self, topic: str, callback: Callable) -> None:
        self.client.message_callback_add(topic, callback)
        self.client.subscribe(topic, qos=1)

    def publish(self, topic: str, payload: str, qos: int = 1) -> None:
        self.client.publish(topic, payload, qos=qos)

    def loop_start(self) -> None:
        self.client.loop_start()


