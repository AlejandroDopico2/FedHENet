# rolann-federated

**Versión:** 1.0.0  
**Licencia:** MIT

Librería de aprendizaje federado que combina:
- **ResNet18 preentrenada** (congelada) como extractor de características.  
- **ROLANN** como clasificador.  
- Cifrado homomórfico opcional vía CKKS (TenSEAL).  
- Comunicación entre clientes y coordinador a través de MQTT.

---

## Instalación

```bash
git clone https://github.com/tu_usuario/rolann-federated.git
cd rolann-federated
pip install .
