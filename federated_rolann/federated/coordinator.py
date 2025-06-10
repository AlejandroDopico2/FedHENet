import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Imports para la comunicación MQTT
import json
import pickle
import base64
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

import tenseal as ts
from ..core import ROLANN

class Coordinador:

    def __init__(self, num_classes, device, num_clients: int, broker: str = "localhost", port: int = 1883, encrypted: bool = False, ctx: ts.Context | None = None):

        # Si encrypted=True pero context contiene clave secreta, falla:
        if encrypted and ctx and ctx.has_secret_key():
            raise ValueError("Pasaste un contexto con clave privada al coordinador")
        
        self.rolann = ROLANN(num_classes=num_classes, encrypted=encrypted, context=ctx)
        self.device = device

        self.M_glb = []  # Matriz M global acumulada para cada clase
        self.U_glb = []  # Matriz U global acumulada para cada clase
        self.S_glb = []  # Matriz S global acumulada para cada clase


        # ResNet preentrenada y congelada
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # resnet propia
        self.resnet.fc = nn.Identity()  # Remplazamos la capa final para extraer características

        for param in self.resnet.parameters():
            param.requires_grad = False  # Congelamos la ResNet

        self.resnet.to(self.device)
        self.resnet.eval()

        # Configuración MQTT
        self.mqtt = mqtt.Client(client_id="coordinator", callback_api_version=CallbackAPIVersion.VERSION1) # mqtt para el coordinador
        self.mqtt.message_callback_add("federated/client/+/update", self._on_client_update) # callback para recibir el modelo de los clientes
        self.mqtt.connect(broker, port) # Conexión al broker MQTT
        self.mqtt.subscribe("federated/client/+/update", qos=1) # Suscripción al tema del modelo de los clientes
        self.mqtt.loop_start() # Inicia el bucle de espera de mensajes

        self.num_clients = num_clients  # Número de clientes
        self._pending = []  # Lista para almacenar los resultados pendientes de los clientes

    # Función para recibir los resultados de los clientes
    def _on_client_update(self, client, userdata, msg):

        data = json.loads(msg.payload) # Deserializa el mensaje recibido
        M_list, US_list = [], []

        
        for i in data: # Recorremos los datos de cada cliente

            m_bytes = base64.b64decode(i["M"]) # Deserializa la matriz M    

            if self.rolann.encrypted:
                try:
                    # reconstruye el vector cifrado sin hacer pickle
                    M_enc = ts.ckks_vector_from(self.rolann.context, m_bytes)
                    M_list.append(M_enc)
                except Exception:
                    # no era ciphertext CKKS: pickle.loads
                    arr = pickle.loads(m_bytes)
                    tensor = torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)
                    M_list.append(tensor)
            else:
                # si no es cifrado, deserializa la matriz M usando pickle
                arr = pickle.loads(m_bytes)
                tensor = torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)
                M_list.append(tensor) 

            # Deserializa US
            US_np = pickle.loads(base64.b64decode(i["US"]))
            US_list.append(torch.from_numpy(US_np).to(self.device))


        self._pending.append((M_list, US_list)) # Almacena los resultados pendientes

        if len(self._pending) == self.num_clients: # Si se han recibido todos los resultados de los clientes
            
            Ms, USs = zip(*self._pending) # Desempaqueta los resultados pendientes
            self.recolect_parcial(list(Ms), list(USs)) # Recolecta los resultados
            self._pending.clear() # Limpia la lista de pendientes

            # Serializar y publicar el modelo global
            body = []
            for M_glb, U_glb, S_glb in zip(self.M_glb, self.U_glb, self.S_glb):

                US_np = (U_glb @ torch.diag(S_glb)).cpu().numpy() # Reconstruimos matriz US
                us_bytes = pickle.dumps(US_np) # Obtenemos la matriz US en bytes

                # M_glb puede ser CKKSVector o tensor
                if hasattr(M_glb, "serialize"):
                    # ciphertext puro –> serialize()
                    m_bytes = M_glb.serialize()
                else:
                    # tensor –> numpy + pickle
                    m_bytes = pickle.dumps(M_glb.cpu().numpy())

                # Guardar en el body el modelo global
                body.append({
                    "M": base64.b64encode(m_bytes).decode(),
                    "US": base64.b64encode(us_bytes).decode(),
                })

            self.mqtt.publish("federated/global_model", json.dumps(body), qos=1) # Publica el modelo global

    def recolect_parcial(self, M_list, US_list):
        """
        Recolecta las matrices M y US de cada cliente y las agrega para formar el modelo global.
        """

        # Number of classes
        nclasses = len(M_list[0])
        init = False

        # For each class the results of each client are aggregated    
        for c in range(0, nclasses):

            if (not self.M_glb) or init:            
                init = True
                M  = M_list[0][c]
                US = US_list[0][c]
                M_rest  = [item[c] for item in M_list[1:]]
                US_rest = [item[c] for item in US_list[1:]]
            else:
                M = self.M_glb[c]
                US = self.U_glb[c] @ np.diag(self.S_glb[c])
                M_rest  = [item[c] for item in M_list[:]]
                US_rest = [item[c] for item in US_list[:]]

            # Aggregation of M and US
            for M_k, US_k in zip(M_rest, US_rest):

                M = M + M_k
                
                # Convertir ambos a tensores en el dispositivo correcto
                if not isinstance(US_k, torch.Tensor):
                    US_k = torch.from_numpy(US_k).to(self.device)
                else:
                    US_k = US_k.to(self.device)

                if not isinstance(US, torch.Tensor):
                    US = torch.from_numpy(US).to(self.device)
                else:
                    US = US.to(self.device)

                # Concatenar en la dimensión de columnas
                concatenated = torch.cat((US_k, US), dim=1)

                # SVD
                U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

                # Multiplicación matricial sin usar @
                US = torch.matmul(U, torch.diag(S))

            # Save contents
            if init:
                self.M_glb.append(M)
                self.U_glb.append(U)
                self.S_glb.append(S)
            else:
                self.M_glb[c] = M
                self.U_glb[c] = U
                self.S_glb[c] = S

        self.update_global(self.M_glb, self.U_glb, self.S_glb)



    def update_global(self, mg_list, ug_list, sg_list):
        """
        Actualiza el modelo global de ROLANN con las matrices globales calculadas.
        """ 

        if self.rolann.encrypted:       
            self.rolann.mg = mg_list # No es tensor, es ckks vector 
        else:
            mg_tensor_list = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).to(self.device) for m in mg_list]
            self.rolann.mg = mg_tensor_list


        ug_tensor_list = [u if isinstance(u, torch.Tensor) else torch.from_numpy(u).to(self.device) for u in ug_list]
        sg_tensor_list = [s if isinstance(s, torch.Tensor) else torch.from_numpy(s).to(self.device) for s in sg_list]
        
        # Asignamos las listas directamente al modelo global
        self.rolann.ug = ug_tensor_list
        self.rolann.sg = sg_tensor_list
        

