#!/usr/bin/env python3
"""
Test script to verify Q4_K packing format compatibility with llama.cpp
"""

def unpack_q4_k_scales_llama_style(scales_bytes):
    """
    Unpack Q4_K scales using llama.cpp's get_scale_min_k4 logic
    """
    sc = [0] * 8
    m = [0] * 8
    
    for j in range(8):
        if j < 4:
            sc[j] = scales_bytes[j] & 63
            m[j] = scales_bytes[j + 4] & 63
        else:
            sc[j] = (scales_bytes[j+4] & 0xF) | ((scales_bytes[j-4] >> 6) << 4)
            m[j] = (scales_bytes[j+4] >>  4) | ((scales_bytes[j-0] >> 6) << 4)
    
    return sc, m

def unpack_q4_k_scales_current(scales_bytes):
    """
    Current C-Kernel-Engine implementation
    """
    sc = [0] * 8
    m = [0] * 8
    
    # Unpack scales (sc[0..7]) from bytes 0-5
    sc[0] = (scales_bytes[0] & 0x3F)
    sc[1] = (scales_bytes[0] >> 6) | ((scales_bytes[1] & 0x0F) << 2)
    sc[2] = (scales_bytes[1] >> 4) | ((scales_bytes[2] & 0x03) << 4)
    sc[3] = (scales_bytes[2] >> 2)
    sc[4] = (scales_bytes[3] & 0x3F)
    sc[5] = (scales_bytes[3] >> 6) | ((scales_bytes[4] & 0x0F) << 2)
    sc[6] = (scales_bytes[4] >> 4) | ((scales_bytes[5] & 0x03) << 4)
    sc[7] = (scales_bytes[5] >> 2)

    # Unpack mins (m[0..7]) from bytes 6-11
    m[0] = (scales_bytes[6] & 0x3F)
    m[1] = (scales_bytes[6] >> 6) | ((scales_bytes[7] & 0x0F) << 2)
    m[2] = (scales_bytes[7] >> 4) | ((scales_bytes[8] & 0x03) << 4)
    m[3] = (scales_bytes[8] >> 2)
    m[4] = (scales_bytes[9] & 0x3F)
    m[5] = (scales_bytes[9] >> 6) | ((scales_bytes[10] & 0x0F) << 2)
    m[6] = (scales_bytes[10] >> 4) | ((scales_bytes[11] & 0x03) << 4)
    m[7] = (scales_bytes[11] >> 2)
    
    return sc, m

def test_packing_compatibility():
    """Test with known values"""
    # Create test data - 12 bytes of scales
    test_scales = bytes([
        0b00111111,  # byte 0: sc[0] = 0x3F, sc[1] bits 0-1
        0b01010101,  # byte 1: sc[1] bits 2-5, sc[2] bits 0-3
        0b10101010,  # byte 2: sc[2] bits 4-5, sc[3]
        0b11001100,  # byte 3: sc[4]
        0b00110011,  # byte 4: sc[5] bits 0-1
        0b01010101,  # byte 5: sc[5] bits 2-5, sc[6] bits 0-3
        0b10101010,  # byte 6: sc[6] bits 4-5, sc[7], m[0]
        0b11001100,  # byte 7: m[1] bits 0-1
        0b00110011,  # byte 8: m[1] bits 2-5, m[2] bits 0-3
        0b01010101,  # byte 9: m[2] bits 4-5, m[3]
        0b10101010,  # byte 10: m[4]
        0b11001100   # byte 11: m[5] bits 0-1
    ])
    
    print("Test scales bytes:", [f"0x{b:02x}" for b in test_scales])
    print()
    
    # Unpack using both methods
    sc_llama, m_llama = unpack_q4_k_scales_llama_style(test_scales)
    sc_current, m_current = unpack_q4_k_scales_current(test_scales)
    
    print("llama.cpp style unpacking:")
    print("Scales:", [f"0x{s:02x}" for s in sc_llama])
    print("Mins:  ", [f"0x{m:02x}" for m in m_llama])
    print()
    
    print("Current C-Kernel-Engine unpacking:")
    print("Scales:", [f"0x{s:02x}" for s in sc_current])
    print("Mins:  ", [f"0x{m:02x}" for m in m_current])
    print()
    
    # Check if they match
    matches = True
    for i in range(8):
        if sc_llama[i] != sc_current[i] or m_llama[i] != m_current[i]:
            matches = False
            print(f"Mismatch at index {i}: llama=({sc_llama[i]:02x}, {m_llama[i]:02x}) vs current=({sc_current[i]:02x}, {m_current[i]:02x})")
    
    if matches:
        print("✅ Packing formats MATCH!")
    else:
        print("❌ Packing formats DO NOT MATCH!")
        print("\nThe current implementation needs to be fixed.")

if __name__ == "__main__":
    test_packing_compatibility()