import tenseal as ts


def create_context(
    poly_modulus_degree: int = 32768,
    coeff_mod_bit_sizes: list[int] = (60, 40, 40, 60),
    global_scale: float = 2**40,
    generate_galois: bool = True,
) -> ts.Context:
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    if generate_galois:
        ctx.generate_galois_keys()
    ctx.global_scale = global_scale
    return ctx


def serialize_context(ctx: ts.Context, secret_key: bool) -> bytes:
    """
    Serialize the context. If secret_key is True, include the secret key
    (for clients). Otherwise, make the context public and serialize
    without the secret key.
    """
    if secret_key:
        return ctx.serialize(save_secret_key=True)
    ctx.make_context_public()
    return ctx.serialize()


def deserialize_context(ctx_bytes: bytes) -> ts.Context:
    """Reconstruct a CKKS context from bytes."""
    return ts.context_from(ctx_bytes)
