try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64

class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, num_clients: int, security_bits: int = 256):
        """Initialize secure aggregation protocol.
        
        Args:
            num_clients: Number of participating clients
            security_bits: Security parameter in bits
        """
        self.num_clients = num_clients
        self.security_bits = security_bits
        self.logger = logging.getLogger(__name__)
        
        # Generate random seeds for masking
        self.random_seeds = {
            i: os.urandom(32) for i in range(num_clients)
        }
        
        # Initialize shared secret keys
        self.shared_keys = {}
        
    def generate_client_keys(self, client_id: int) -> Dict[str, bytes]:
        """Generate cryptographic keys for a client."""
        # Generate client's secret key
        secret_key = os.urandom(32)
        
        # Generate pairwise keys with other clients
        pairwise_keys = {
            j: self._generate_pairwise_key(secret_key, self.random_seeds[j])
            for j in range(self.num_clients)
            if j != client_id
        }
        
        return {
            'secret_key': secret_key,
            'pairwise_keys': pairwise_keys
        }
        
    def _generate_pairwise_key(self, key1: bytes, key2: bytes) -> bytes:
        """Generate a shared key between two clients."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'federated_learning',
            iterations=100000,
        )
        combined = bytes([a ^ b for a, b in zip(key1, key2)])
        return base64.urlsafe_b64encode(kdf.derive(combined))

    def mask_weights(self, 
                    weights: List[np.ndarray],
                    client_id: int,
                    round_number: int) -> List[np.ndarray]:
        """Apply masking to weights using pairwise random values."""
        try:
            masked_weights = []
            for layer in weights:
                # Generate deterministic mask using pairwise keys
                mask = np.zeros_like(layer)
                for j, key in self.shared_keys.items():
                    if j != client_id:
                        # Use key and round number to generate deterministic random mask
                        rng = np.random.RandomState(
                            int.from_bytes(key + str(round_number).encode(), 'big')
                        )
                        mask += rng.normal(0, 1, layer.shape)
                
                # Add mask to weights
                masked_weights.append(layer + mask)
                
            return masked_weights
            
        except Exception as e:
            self.logger.error(f"Error masking weights: {str(e)}")
            return weights

    def unmask_aggregated_weights(self,
                                masked_weights: List[np.ndarray],
                                round_number: int) -> List[np.ndarray]:
        """Remove masks from aggregated weights."""
        try:
            unmasked_weights = []
            for layer in masked_weights:
                # Remove all masks
                total_mask = np.zeros_like(layer)
                for client_id in range(self.num_clients):
                    for j, key in self.shared_keys.items():
                        if j != client_id:
                            rng = np.random.RandomState(
                                int.from_bytes(key + str(round_number).encode(), 'big')
                            )
                            total_mask += rng.normal(0, 1, layer.shape)
                
                # Remove total mask from aggregated weights
                unmasked_weights.append(layer - total_mask)
                
            return unmasked_weights
            
        except Exception as e:
            self.logger.error(f"Error unmasking weights: {str(e)}")
            return masked_weights

class EnhancedEncryption:
    """Enhanced encryption for federated learning."""
    
    def __init__(self, 
                 security_level: str = 'high',
                 enable_secure_aggregation: bool = True):
        """Initialize enhanced encryption.
        
        Args:
            security_level: 'medium' or 'high'
            enable_secure_aggregation: Whether to use secure aggregation
        """
        self.logger = logging.getLogger(__name__)
        self.security_level = security_level
        self.enable_secure_aggregation = enable_secure_aggregation
        
        # Initialize TENSEAL context with appropriate parameters
        self.context = self._create_context()
        
        # Generate encryption key for additional layer of security
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
        # Initialize secure aggregation if enabled
        self.secure_agg = None
        if enable_secure_aggregation:
            self.secure_agg = SecureAggregation(num_clients=2)  # Default to 2 clients

    def _create_context(self):
        """Create TENSEAL context with appropriate security parameters."""
        if not TENSEAL_AVAILABLE:
            self.logger.warning("TenSeal not available, using basic encryption")
            return None
            
        if self.security_level == 'high':
            # Higher security parameters
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=65536,
                coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
            )
        else:
            # Medium security parameters
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=32768,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            
        context.generate_galois_keys()
        context.global_scale = 2**40
        return context

    def encrypt_weights(self, 
                       weights: List[np.ndarray],
                       client_id: Optional[int] = None,
                       round_number: Optional[int] = None) -> List[bytes]:
        """Encrypt model weights with enhanced security."""
        try:
            # Apply secure aggregation masking if enabled
            if self.enable_secure_aggregation and client_id is not None:
                weights = self.secure_agg.mask_weights(
                    weights, client_id, round_number or 0
                )
            
            encrypted_weights = []
            import json
            
            for weight in weights:
                # Store shape information
                original_shape = weight.shape
                # Flatten and convert to list
                weight_array = weight.flatten().tolist()
                
                # Create data package with shape
                data_package = {
                    'shape': original_shape,
                    'data': weight_array
                }
                
                if TENSEAL_AVAILABLE and self.context is not None:
                    # Encrypt with TENSEAL if available
                    encrypted_vector = ts.ckks_vector(self.context, weight_array)
                    encrypted_data = encrypted_vector.serialize()
                else:
                    # Fallback to JSON serialization
                    encrypted_data = json.dumps(data_package).encode()
                
                # Add additional layer of encryption
                encrypted_data = self.cipher.encrypt(encrypted_data)
                encrypted_weights.append(encrypted_data)
                
            return encrypted_weights
            
        except Exception as e:
            self.logger.error(f"Error encrypting weights: {str(e)}")
            raise

    def decrypt_weights(self, 
                       encrypted_weights: List[bytes],
                       unmask: bool = True) -> List[np.ndarray]:
        """Decrypt model weights."""
        try:
            decrypted_weights = []
            import json
            
            for enc_weight in encrypted_weights:
                # Decrypt outer layer
                decrypted_data = self.cipher.decrypt(enc_weight)
                
                if TENSEAL_AVAILABLE and self.context is not None:
                    # Decrypt TENSEAL layer if available
                    enc_vector = ts.ckks_vector_from(self.context, decrypted_data)
                    weight_array = np.array(enc_vector.decrypt())
                else:
                    # Fallback from JSON
                    data_package = json.loads(decrypted_data.decode())
                    weight_array = np.array(data_package['data'])
                    original_shape = tuple(data_package['shape'])
                    weight_array = weight_array.reshape(original_shape)
                
                decrypted_weights.append(weight_array)
                
            # Remove secure aggregation masks if enabled
            if self.enable_secure_aggregation and unmask:
                decrypted_weights = self.secure_agg.unmask_aggregated_weights(
                    decrypted_weights, round_number=0
                )
                
            return decrypted_weights
            
        except Exception as e:
            self.logger.error(f"Error decrypting weights: {str(e)}")
            raise

    def setup_secure_aggregation(self, num_clients: int):
        """Set up secure aggregation with specified number of clients."""
        if self.enable_secure_aggregation:
            self.secure_agg = SecureAggregation(num_clients)

# For backwards compatibility
def create_context():
    """Create default encryption context."""
    encryption = EnhancedEncryption()
    return encryption

def encrypt_weights(context, model_weights):
    """Encrypt weights using default context."""
    if isinstance(context, EnhancedEncryption):
        return context.encrypt_weights(model_weights)
    else:
        # Legacy encryption
        encryption = EnhancedEncryption()
        return encryption.encrypt_weights(model_weights)

def decrypt_weights(context, encrypted_weights):
    """Decrypt weights using default context."""
    if isinstance(context, EnhancedEncryption):
        return context.decrypt_weights(encrypted_weights)
    else:
        # Legacy decryption
        encryption = EnhancedEncryption()
        return encryption.decrypt_weights(encrypted_weights)
