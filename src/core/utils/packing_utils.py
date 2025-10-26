import struct
from typing import List, Tuple


def pack_coordinates(coords: List[Tuple[float, float]]) -> bytes:
    """
    Encode n (x, y) pairs into 8n bytes (2n × uint32).
    Each coordinate is scaled and stored as a 4-byte unsigned integer (uint32).

    Args:
        coords: List of n tuples (x, y). E.g., [(999.99, 999.99), ...]

    Returns:
        bytes: 8n bytes representing the 2n scaled coordinates.
    """
    scaled = []
    for x, y in coords:
        # Scale by 100 and convert to integer (e.g., 999.99 -> 99999).
        x_scaled = int(round(x * 100))
        y_scaled = int(round(y * 100))
        scaled.extend([x_scaled, y_scaled])

    # Pack the 2n uint32 integers (4 bytes each) into 8n bytes.
    nb_coords = 2 * len(coords)
    format_string = '>' + str(nb_coords) + 'I'
    # 'I' = uint32 (4 bytes, little-endian)
    return struct.pack(format_string, *scaled)


def unpack_coordinates(packed_bytes: bytes) -> List[Tuple[float, float]]:
    """
    Decode the 40 bytes (10 uint32 integers) into 5 (x, y) pairs.
    This function is hardcoded to unpack exactly 10 uint32 values (40 bytes).
    """
    # Unpack 10 uint32 integers (4 bytes each), totaling 40 bytes.
    scaled = struct.unpack('>10I', packed_bytes)

    # Reconstruct the 5 (x, y) pairs
    coords = []
    # Loop over the 10 scaled values in steps of 2
    for i in range(0, 10, 2):
        x_scaled, y_scaled = scaled[i], scaled[i+1]
        # Rescale back to float
        x = x_scaled / 100
        y = y_scaled / 100
        coords.append((x, y))
    return coords
