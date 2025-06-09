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
cd rolann_federated
pip install -e .
```
---

## Configuración

Todos los parámetros principales se pueden ajustar:

| Parámetro         | Descripción                                         | Valor por defecto  |
|-------------------|-----------------------------------------------------|--------------------|
| `broker`          | Dirección del broker MQTT                           | `"localhost"`      |
| `port`            | Puerto del broker MQTT                              | `1883`             |
| `num_clients`     | Número de clientes a instanciar                     | `_`                |
| `num_classes`     | Número de clases del dataset                        | `-`                |
| `device`          | Dispositivo PyTorch (`"cpu"` o `"cuda"`)            | `"cpu"`            |
| `encrypted`       | Activar cifrado homomórfico (True/False)            | `False`            |

Además, los parámetros específicos de CKKS (TenSEAL) se crean en el propio script:

- `poly_modulus_degree`: Tamaño del polinomio (p.ej. 8192).
- `coeff_mod_bit_sizes`: Tamaño de coeficientes (p.ej. `[60]`).
- `global_scale`: Escala global (p.ej. `2**40`).

Puedes modificar estos valores directamente en el bloque de creación de contexto.

---

## Ejemplo de uso

1. **Levanta un broker MQTT** (por ejemplo Mosquitto):
   ```bash
   # Local:
   pip install mosquitto
   ```

2. **Ejecuta el ejemplo** `example.py` (ya incluido en el paquete):
   ```bash
   python example.py
   ```
   Este script:
   - Crea un contexto CKKS con clave privada y pública.
   - Instancia 1 coordinador y 2 clientes con cifrado.
   - Realiza un entrenamiento local y envía las actualizaciones.
   - El coordinador agrega automáticamente y publica el modelo global.
   - Se imprime por pantalla la precisión final del modelo
