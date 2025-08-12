import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm
import numpy as np

# Imports for MQTT communication
import json
import pickle
import base64
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
from ..core import ROLANN
import tenseal as ts

class Client:
    def __init__(self, num_classes, dataset, device, client_id: int, broker: str = "localhost", port: int = 1883, encrypted: bool = False, ctx: ts.Context | None = None):

        # If encrypted=True but context does NOT have a secret key, fail:
        if encrypted and (ctx is None or not ctx.has_secret_key()):
            raise ValueError("For the client, context must include a private key")
        
        self.device = device # Device (CPU or GPU) where the client will run
        self.rolann = ROLANN(num_classes=num_classes, encrypted=encrypted, context=ctx) # Instance of the ROLANN class
        self.loader = DataLoader(dataset, batch_size=128, shuffle=True) # Local dataset

        # Each client creates its own pretrained and frozen ResNet
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT) # Own resnet
        self.resnet.fc = nn.Identity()  # Replace the final layer to extract features

        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze the ResNet

        self.resnet.to(self.device) # Move ResNet to device
        self.resnet.eval()
        self.rolann.to(self.device)  # Ensure ROLANN is on the same device

        # MQTT configuration 
        self.mqtt = mqtt.Client(client_id=f"client_{client_id}", callback_api_version=CallbackAPIVersion.VERSION1) # mqtt for each client
        self.mqtt.message_callback_add("federated/global_model", self._on_global_model) # callback to receive the global model
        self.mqtt.connect(broker, port) # Connect to the MQTT broker
        self.mqtt.subscribe("federated/global_model", qos=1) # Subscribe to the global model topic
        self.mqtt.loop_start() # Start the message loop

        self.client_id = client_id  # Client ID

    
    def training(self):
        """
        Iterate over the local dataset, extract features using the local ResNet and
        update the ROLANN layer
        """
        self.resnet.to(self.device) # Move to training and move to cpu after training

        for x, y in tqdm(self.loader):

            x = x.to(self.device)

            with torch.no_grad():
                features = self.resnet(x)  # Extract local features

            # Convert labels to one-hot to match the number of classes
            label = (torch.nn.functional.one_hot(y, num_classes=10) * 0.9 + 0.05).to(self.device)
            self.rolann.aggregate_update(features, label)

        # Move resnet to cpu
        self.resnet.to("cpu")

    def aggregate_parcial(self):
        """
        Publish the local model to the MQTT broker
        """
        # Return the accumulated matrices M and US for each class
        local_M = self.rolann.mg
        local_US = [torch.matmul(self.rolann.ug[i], torch.diag(self.rolann.sg[i].clone().detach())) for i in range(self.rolann.num_classes)]


        # Serialize and publish the update
        body = [] # Create a body for the message

        for M_enc, US in zip(local_M, local_US): # Iterate over the accumulated matrices

            # If CKKSVector, serialize, otherwise convert to list
            if hasattr(M_enc, "serialize"):
                serialized = M_enc.serialize()
                bM = base64.b64encode(serialized).decode()
            else:
                # tensor
                m_plain = M_enc.cpu().numpy().tolist()
                bM = base64.b64encode(pickle.dumps(m_plain)).decode()

            # Serialize US and add to body
            bUS = base64.b64encode(pickle.dumps(US.cpu().numpy())).decode() # bUS is the serialized US matrix, i.e., the US matrix in bytes
            body.append({"M": bM, "US": bUS}) # Add to the body the dictionary with M and US matrices

        topic = f"federated/client/{self.client_id}/update" # Create the topic for the client
        self.mqtt.publish(topic, json.dumps(body), qos=1) # Publish the message to the topic


    # Receives the global model and decomposes it into M and US matrices
    def _on_global_model(self, client, userdata, msg):

        data = json.loads(msg.payload) # Deserialize the received message
        mg, ug, sg = [], [], []
        for i in data: # Iterate over the received data

            m_bytes = base64.b64decode(i["M"])

            # if CKKS ciphertext, reconstruct, otherwise pickle
            try:
                M_enc = ts.ckks_vector_from(self.rolann.context, m_bytes)
                mg.append(M_enc)
            except Exception:
                arr = pickle.loads(m_bytes)
                mg.append(torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device))

            US_np = pickle.loads(base64.b64decode(i["US"])) # Deserialize the US matrix

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



    def evaluate(self, loader): # Add the ResNet18 model
        correct = 0
        total = 0

        self.resnet.to(self.device)
        self.rolann.to(self.device)

        with torch.no_grad():
            for x, y in loader:

                x = x.to(self.device) # Move data to GPU
                y = y.to(self.device) # Move labels to GPU

                caracterisiticas = self.resnet(x) # Get features from ResNet18
                preds = self.rolann(caracterisiticas) # Get predictions from ROLANN

                correct += (preds.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        return correct / total
