import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# Imports for MQTT communication
import json
import pickle
import base64
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion

import tenseal as ts
from ..core import ROLANN

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
        rolann: ROLANN | None = None,
    ):
        
        # --- Optional ROLANN injection ---
        if rolann is not None:
            # Ensure num_classes consistency
            if hasattr(rolann, "num_classes") and rolann.num_classes != num_classes:
                raise ValueError(
                    f"Inconsistent num_classes: constructor={num_classes} vs rolann.num_classes={rolann.num_classes}"
                )
            # The coordinator must NOT hold a secret key if encryption is enabled
            if getattr(rolann, "encrypted", False):
                rctx = getattr(rolann, "context", None)
                if rctx is not None and rctx.has_secret_key():
                    raise ValueError(
                        "Coordinator must not hold a secret key in the CKKS context"
                    )
            self.rolann = rolann
        else:
            # Default mode: create ROLANN instance using constructor flags
            if encrypted and ctx and ctx.has_secret_key():
                raise ValueError("You passed a context with a private key to the coordinator")
            self.rolann = ROLANN(num_classes=num_classes, encrypted=encrypted, context=ctx)

        self.device = device

        self.M_global = []  # Global M matrix accumulated for each class
        self.U_global = []  # Global U matrix accumulated for each class
        self.S_global = []  # Global S matrix accumulated for each class

        # Pretrained and frozen ResNet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Replace the final layer to extract features

        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze the ResNet

        self.resnet.to(self.device)
        self.resnet.eval()

        # MQTT configuration
        self.mqtt = mqtt.Client(client_id="coordinator", callback_api_version=CallbackAPIVersion.VERSION1)
        self.mqtt.message_callback_add("federated/client/+/update", self._on_client_update)  # Callback to receive client models
        self.mqtt.connect(broker, port)  # Connect to MQTT broker
        self.mqtt.subscribe("federated/client/+/update", qos=1)  # Subscribe to client model topic
        self.mqtt.loop_start()  # Start message loop

        self.num_clients = num_clients  # Number of clients
        self._pending = []  # List to store pending results from clients

    # Function to receive results from clients
    def _on_client_update(self, client, userdata, msg):

        data = json.loads(msg.payload)  # Deserialize received message
        M_list, US_list = [], []

        for i in data:  # Iterate over each client's data

            m_bytes = base64.b64decode(i["M"])  # Deserialize M matrix    

            if self.rolann.encrypted:
                try:
                    # Rebuild encrypted vector without pickle
                    M_enc = ts.ckks_vector_from(self.rolann.context, m_bytes)
                    M_list.append(M_enc)
                except Exception:
                    # Not a CKKS ciphertext: use pickle.loads
                    arr = pickle.loads(m_bytes)
                    tensor = torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)
                    M_list.append(tensor)
            else:
                # If not encrypted, deserialize M matrix using pickle
                arr = pickle.loads(m_bytes)
                tensor = torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device)
                M_list.append(tensor) 

            # Deserialize US
            US_np = pickle.loads(base64.b64decode(i["US"]))
            US_list.append(torch.from_numpy(US_np).to(self.device))

        self._pending.append((M_list, US_list))  # Store pending results

        if len(self._pending) == self.num_clients:  # If all client results have been received
            
            Ms, USs = zip(*self._pending)  # Unpack pending results
            self.partial_collect(list(Ms), list(USs))  # Aggregate results
            self._pending.clear()  # Clear pending list

            # Serialize and publish global model
            body = []
            for M_global, U_global, S_global in zip(self.M_global, self.U_global, self.S_global):

                US_np = (U_global @ torch.diag(S_global)).cpu().numpy()  # Reconstruct US matrix
                us_bytes = pickle.dumps(US_np)  # Get US matrix as bytes

                # M_global can be CKKSVector or tensor
                if hasattr(M_global, "serialize"):
                    # Pure ciphertext –> serialize()
                    m_bytes = M_global.serialize()
                else:
                    # tensor –> numpy + pickle
                    m_bytes = pickle.dumps(M_global.cpu().numpy())

                # Save global model in body
                body.append({
                    "M": base64.b64encode(m_bytes).decode(),
                    "US": base64.b64encode(us_bytes).decode(),
                })

            self.mqtt.publish("federated/global_model", json.dumps(body), qos=1)  # Publish global model

    def partial_collect(self, M_list, US_list):
        """
        Collects the M and US matrices from each client and aggregates them to form the global model.
        """

        nclasses = len(M_list[0])
        init = False

        # For each class, aggregate the results from each client    
        for c in range(0, nclasses):

            if (not self.M_global) or init:            
                init = True
                M  = M_list[0][c]
                US = US_list[0][c]
                M_rest  = [item[c] for item in M_list[1:]]
                US_rest = [item[c] for item in US_list[1:]]
            else:
                M = self.M_global[c]
                US = self.U_global[c] @ np.diag(self.S_global[c])
                M_rest  = [item[c] for item in M_list[:]]
                US_rest = [item[c] for item in US_list[:]]

            # Aggregate M and US
            for M_k, US_k in zip(M_rest, US_rest):

                M = M + M_k
                
                # Convert both to tensors on the correct device
                if not isinstance(US_k, torch.Tensor):
                    US_k = torch.from_numpy(US_k).to(self.device)
                else:
                    US_k = US_k.to(self.device)

                if not isinstance(US, torch.Tensor):
                    US = torch.from_numpy(US).to(self.device)
                else:
                    US = US.to(self.device)

                # Concatenate along columns
                concatenated = torch.cat((US_k, US), dim=1)

                # SVD
                U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

                # Matrix multiplication without using @
                US = torch.matmul(U, torch.diag(S))

            # Save contents
            if init:
                self.M_global.append(M)
                self.U_global.append(U)
                self.S_global.append(S)
            else:
                self.M_global[c] = M
                self.U_global[c] = U
                self.S_global[c] = S

        self.update_global(self.M_global, self.U_global, self.S_global)

    def update_global(self, mg_list, ug_list, sg_list):
        """
        Updates the global ROLANN model with the calculated global matrices.
        """ 

        if self.rolann.encrypted:       
            self.rolann.mg = mg_list # Not tensor, is ckks vector 
        else:
            mg_tensor_list = [m if isinstance(m, torch.Tensor) else torch.from_numpy(m).to(self.device) for m in mg_list]
            self.rolann.mg = mg_tensor_list

        ug_tensor_list = [u if isinstance(u, torch.Tensor) else torch.from_numpy(u).to(self.device) for u in ug_list]
        sg_tensor_list = [s if isinstance(s, torch.Tensor) else torch.from_numpy(s).to(self.device) for s in sg_list]
        
        # Assign lists directly to the global model
        self.rolann.ug = ug_tensor_list
        self.rolann.sg = sg_tensor_list
