import tenseal as ts

def create_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_weights(context, model_weights):
    encrypted_weights = []
    for weight in model_weights:
        weight_array = weight.flatten().tolist()
        encrypted_vector = ts.ckks_vector(context, weight_array)
        encrypted_weights.append(encrypted_vector.serialize())
    return encrypted_weights

def decrypt_weights(context, encrypted_weights):
    decrypted_weights = []
    for enc_weight in encrypted_weights:
        enc_vector = ts.ckks_vector_from(context, enc_weight)
        decrypted_weights.append(enc_vector.decrypt())
    return decrypted_weights
