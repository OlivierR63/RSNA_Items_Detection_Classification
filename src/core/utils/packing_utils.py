import struct

def pack_coordinates(coords: list[tuple[float, float]]) -> bytes:
    """Encode n paires (x, y) en 20n octets (2n × uint32, sachant que chaque uint32 est codé sur 4 octets).

    Args:
        coords: Liste de n tuples (x, y). Ex: [(999.99, 999.99), ...]

    Returns:
        bytes: 20n octets représentant les 2n coordonnées scalées.
    """
    scaled = []
    for x, y in coords:
        x_scaled = int(round(x * 100))  # 999.99 → 99999 (4 octets suffisent)
        y_scaled = int(round(y * 100))  # 999.99 → 99999
        scaled.extend([x_scaled, y_scaled])

    # Pack les 10 entiers uint32 (4 octets chacun) en 40 octets
    nb_coords = 2*len(coords)
    format_string = '>' + str(nb_coords) + 'I'
    return struct.pack(format_string, *scaled)  # 'I' = uint32 (4 octets)

def unpack_coordinates(packed_bytes: bytes) -> list[tuple[float, float]]:
    """Décode les 40 octets en 5 paires (x, y)."""
    scaled = struct.unpack('>10I', packed_bytes)  # 10 × uint32

    # Reconstruit les paires (x, y)
    coords = []
    for i in range(0, 10, 2):
        x_scaled, y_scaled = scaled[i], scaled[i+1]
        x = x_scaled / 100
        y = y_scaled / 100
        coords.append((x, y))
    return coords