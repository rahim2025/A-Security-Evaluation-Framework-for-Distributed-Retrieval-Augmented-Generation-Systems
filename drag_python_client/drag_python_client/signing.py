from typing import Union

from eth_account import Account
from eth_account.messages import encode_defunct


def sign_message_personal(message: str, private_key_hex: Union[str, bytes]) -> bytes:
    """
    Produce an off-chain signature compatible with the DragScores contract verification:
    bytes32 ethSignedMessageHash = MessageHashUtils.toEthSignedMessageHash(bytes(message));
    address recovered = ECDSA.recover(ethSignedMessageHash, signature);

    This function uses personal_sign semantics via encode_defunct(text=message).
    Returns 65-byte signature (r||s||v), suitable to pass to the contract.
    """
    if isinstance(private_key_hex, bytes):
        private_key_hex = private_key_hex.hex()
    msg = encode_defunct(text=message)
    signed = Account.sign_message(msg, private_key=private_key_hex)
    # signed.signature is bytes, length 65, v is 27/28 which matches ECDSA.recover expectations
    return signed.signature


