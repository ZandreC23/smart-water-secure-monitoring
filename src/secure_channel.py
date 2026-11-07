"""
secure_channel.py
-----------------
The secure communication half of the framework. Each sensor reading is
encrypted and authenticated before "transmission" and verified on arrival,
protecting confidentiality and integrity in transit, independently of the
anomaly-detection model.

Implementation increment (Testing phase): a replay-protection layer was added.
The Design Phase evaluation identified replay as the weakest case for the
detection models (about 20% caught). Replay is now mitigated at the channel
itself: every packet carries a unique nonce and a timestamp, and the receiver
rejects any packet whose nonce has already been seen or whose timestamp falls
outside a freshness window. This closes the replay gap at the communication
layer rather than relying on the model.

For portability this uses the `cryptography` library's Fernet (AES-128 in CBC
mode with an HMAC-SHA256 authentication tag). On a constrained device a lighter
cipher such as PRESENT or SPECK would be substituted behind the same interface.
"""

import json
import time
import uuid
import hmac
import hashlib
from cryptography.fernet import Fernet, InvalidToken


class SecureChannel:
    def __init__(self, key=None, auth_key=b"shared-node-secret",
                 freshness_window=30.0):
        # Symmetric key shared between the sensor node and the server.
        self.key = key or Fernet.generate_key()
        self._cipher = Fernet(self.key)
        self.auth_key = auth_key
        # Replay protection state.
        self.freshness_window = freshness_window   # seconds a packet stays valid
        self._seen_nonces = set()                  # nonces already accepted

    def send(self, reading: dict) -> bytes:
        """Encrypt and authenticate a reading, stamping it with nonce + time."""
        envelope = {
            "reading": reading,
            "nonce": uuid.uuid4().hex,     # unique per message
            "ts": time.time(),             # send timestamp
        }
        payload = json.dumps(envelope, sort_keys=True).encode()
        token = self._cipher.encrypt(payload)                            # confidentiality
        tag = hmac.new(self.auth_key, token, hashlib.sha256).hexdigest() # integrity
        return token + b"||" + tag.encode()

    def receive(self, packet: bytes) -> dict:
        """
        Verify integrity, decrypt, then enforce freshness and nonce uniqueness.
        Raises ValueError on tampering, stale packets, or replays.
        """
        token, tag = packet.rsplit(b"||", 1)

        # 1. Integrity / authentication.
        expected = hmac.new(self.auth_key, token, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected.encode(), tag):
            raise ValueError("Authentication failed: message was tampered with.")

        # 2. Confidentiality.
        try:
            payload = self._cipher.decrypt(token)
        except InvalidToken:
            raise ValueError("Decryption failed: invalid or corrupted ciphertext.")
        envelope = json.loads(payload.decode())

        # 3. Freshness: reject packets older than the window (stale/replayed).
        age = time.time() - envelope.get("ts", 0)
        if age > self.freshness_window:
            raise ValueError(f"Replay rejected: packet is stale ({age:.1f}s old).")

        # 4. Nonce uniqueness: reject any packet already seen (replayed).
        nonce = envelope.get("nonce")
        if nonce in self._seen_nonces:
            raise ValueError("Replay rejected: nonce already used.")
        self._seen_nonces.add(nonce)

        return envelope["reading"]


def _demo():
    """Show a normal exchange, a tampering attempt, and a replay attempt."""
    ch = SecureChannel()
    reading = {"station": "station1", "PH": 6.8, "DO": 9.1, "TEMP": 27.4}

    packet = ch.send(reading)
    print("Encrypted packet (truncated):", packet[:48], "...")
    print("Received & verified        :", ch.receive(packet))

    # Tampering: flip a byte in transit.
    tampered = bytearray(packet)
    tampered[10] ^= 0x01
    try:
        ch.receive(bytes(tampered))
    except ValueError as e:
        print("Tampering attempt          :", e)

    # Replay: resend the exact same (valid) packet.
    try:
        ch.receive(packet)
    except ValueError as e:
        print("Replay attempt             :", e)


if __name__ == "__main__":
    _demo()
