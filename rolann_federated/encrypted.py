import tenseal as ts

def create_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: list[int] = (60,),
    global_scale: float = 2**40,
    generate_galois: bool = True,
) -> ts.Context:
    """
    Crea un contexto CKKS de TenSEAL.
    """
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    if generate_galois:
        ctx.generate_galois_keys()
    ctx.global_scale = global_scale
    return ctx


def serialize_context(ctx, secret_key: bool) -> bytes:
    """
    Serializa el contexto CKKS. 
    - secret_key=True incluye la clave secreta (para clientes).
    - secret_key=False solo deja la parte pública (para coordinador).
    """
    return ctx.serialize(secret_key=secret_key)

def deserialize_context(ctx_bytes: bytes) -> ts.Context:
    """
    Reconstruye un contexto CKKS a partir de bytes (serializado con o sin clave secreta).
    """
    return ts.context_from(ctx_bytes)
