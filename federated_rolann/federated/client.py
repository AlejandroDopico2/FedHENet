import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np

# Imports para la comunicación MQTT
import json
import pickle
import base64
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
from ..core import ROLANN  # Asegúrate de que la ruta es correcta
import tenseal as ts

class Cliente:
    def __init__(self, num_classes, dataset, device, client_id: int, broker: str = "localhost", port: int = 1883, encrypted: bool = False, ctx: ts.Context | None = None):

        # Si encrypted=True pero context NO tiene clave secreta, falla:
        if encrypted and (ctx is None or not ctx.has_secret_key()):
            raise ValueError("Para el cliente, context debe incluir clave privada")
        
        self.device = device # Dispositivo (CPU o GPU) donde se ejecutará el cliente
        self.rolann = ROLANN(num_classes=num_classes, encrypted=encrypted, context=ctx) # Instancia de la clase ROLANN
        self.loader = DataLoader(dataset, batch_size=128, shuffle=True) # dataset local

        # Cada cliente crea su propia ResNet preentrenada y congelada
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # resnet propia
        self.resnet.fc = nn.Identity()  # Remplazamos la capa final para extraer características

        for param in self.resnet.parameters():
            param.requires_grad = False  # Congelamos la ResNet

        self.resnet.to(self.device) # Mover la ResNet al dispositivo
        self.resnet.eval()
        self.rolann.to(self.device)  # Aseguramos que ROLANN esté en el mismo dispositivo

        # Configuración MQTT 
        self.mqtt = mqtt.Client(client_id=f"client_{client_id}", callback_api_version=CallbackAPIVersion.VERSION1) # mqtt para cada cliente
        self.mqtt.message_callback_add("federated/global_model", self._on_global_model) # callback para recibir el modelo global
        self.mqtt.connect(broker, port) # Conexión al broker MQTT
        self.mqtt.subscribe("federated/global_model", qos=1) # Suscripción al tema del modelo global
        self.mqtt.loop_start() # Inicia el bucle de espera de mensajes

        self.client_id = client_id  # ID del cliente

    
    def training(self):
        """
        Recorre el dataset local, extrae las características usando la ResNet propia y
        actualiza la capa ROLANN
        """
        self.resnet.to(self.device) # Mover al training y pasar a cpu al terminar training

        for x, y in tqdm(self.loader):

            x = x.to(self.device)

            with torch.no_grad():
                features = self.resnet(x)  # Extraemos características locales

            # Convertimos las etiquetas a one-hot para que coincidan con el número de clases
            label = (torch.nn.functional.one_hot(y, num_classes=10) * 0.9 + 0.05).to(self.device)
            self.rolann.aggregate_update(features, label)

        # Movemos la resnet a cpu
        self.resnet.to("cpu")

    def aggregate_parcial(self):
        """
        Publica el modelo local en el broker MQTT
        """
        # Devuelve las matrices acumuladas M y US para cada clase
        local_M = self.rolann.mg
        local_US = [torch.matmul(self.rolann.ug[i], torch.diag(self.rolann.sg[i].clone().detach())) for i in range(self.rolann.num_classes)]


        # Serializar y publicar la actualización
        body = [] # Creamos un cuerpo para el mensaje

        for M_enc, US in zip(local_M, local_US): # Recorremos las matrices acumuladas

            # Si es CKKSVector, serializamos, si no, lo convertimos a lista
            if hasattr(M_enc, "serialize"):
                serialized = M_enc.serialize()
                bM = base64.b64encode(serialized).decode()
            else:
                # tensor
                m_plain = M_enc.cpu().numpy().tolist()
                bM = base64.b64encode(pickle.dumps(m_plain)).decode()

            # Serializar US y añadir al cuerpo
            bUS = base64.b64encode(pickle.dumps(US.cpu().numpy())).decode() # bUS es la matriz US serializada es decir la matriz US en bytes
            body.append({"M": bM, "US": bUS}) # Añadimos al cuerpo el diccionario con la matriz M y US

        topic = f"federated/client/{self.client_id}/update" # Creamos el topic para el cliente
        self.mqtt.publish(topic, json.dumps(body), qos=1) # Publicamos el mensaje en el topic


    # Recibe el modelo global y lo descompone en matrices M y US
    def _on_global_model(self, client, userdata, msg):

        data = json.loads(msg.payload) # Deserializa el mensaje recibido
        mg, ug, sg = [], [], []
        for i in data: # Recorre los datos recibidos


            m_bytes = base64.b64decode(i["M"])

            # si es ciphertext CKKS, lo reconstruimos, si no, pickle
            try:
                M_enc = ts.ckks_vector_from(self.rolann.context, m_bytes)
                mg.append(M_enc)
            except Exception:
                arr = pickle.loads(m_bytes)
                mg.append(torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device))

            US_np = pickle.loads(base64.b64decode(i["US"])) # Deserializa la matriz US

            # Descomponemos US en U y S
            U, S, _ = torch.linalg.svd(
                torch.from_numpy(US_np).to(self.device), full_matrices=False
            ) 
            ug.append(U)
            sg.append(S)

        # Actualizamos las matrices acumuladas de ROLANN    
        self.rolann.mg = mg
        self.rolann.ug = ug
        self.rolann.sg = sg
        self.rolann._calculate_weights()



    def evaluate(self, loader): # Añadimos el modelo de ResNet18
        correct = 0
        total = 0

        self.resnet.to(self.device)
        self.rolann.to(self.device)

        with torch.no_grad():
            for x, y in loader:

                x = x.to(self.device) # Subimos los datos a la GPU
                y = y.to(self.device) # Subimos las etiquetas a la GPU

                caracterisiticas = self.resnet(x) # Obtenemos las características de la ResNet18
                preds = self.rolann(caracterisiticas) # Obtenemos las predicciones de la ROLANNs

                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / total