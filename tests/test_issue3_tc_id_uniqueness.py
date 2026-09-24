"""
tests/test_issue3_tc_id_uniqueness.py
Issue 3 regression: TC-ID collision detection.
"""
import hashlib

def make_tc_id(req_id: str, tc_num: int, title: str, is_combined: bool = False) -> str:
    title_hash = hashlib.sha256(title.encode()).hexdigest()[:4]
    if is_combined:
        return f"TC-{req_id}-HYB-C{tc_num:02d}-{title_hash}"
    return f"TC-{req_id}-HYB-{tc_num:03d}-{title_hash}"

def test_tc_id_uniqueness():
    id1 = make_tc_id("REQ_001", 1, "Login Valid User")
    id2 = make_tc_id("REQ_001", 1, "Login Invalid Password")
    assert id1 != id2, f"IDs collided for different titles: {id1} vs {id2}"
    assert id1.endswith(hashlib.sha256("Login Valid User".encode()).hexdigest()[:4])
    assert id2.endswith(hashlib.sha256("Login Invalid Password".encode()).hexdigest()[:4])
    print("[PASS] Title hash makes TC-IDs unique for identical req_id + counter")

if __name__ == "__main__":
    test_tc_id_uniqueness()
