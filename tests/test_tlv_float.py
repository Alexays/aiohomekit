from dataclasses import dataclass
from aiohomekit.tlv8 import TLVStruct, tlv_entry, u8

@dataclass
class HasFloat(TLVStruct):
    a: u8 = tlv_entry(1)
    f: float = tlv_entry(2)

def test_float_round_trip():
    v = HasFloat(a=3, f=0.5)
    out = HasFloat.decode(v.encode())
    assert out.a == 3
    assert abs(out.f - 0.5) < 1e-6
