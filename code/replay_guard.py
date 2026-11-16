"""Verify historical source/selection records before decoding a replay case."""
import hashlib
import json


# Supplemental audit, added after scoring: original raw hashes matched the old
# freezes before deriving these LF equivalents. Historical freezes are unchanged.
MANIFEST_HASHES = {
    "reserve": ("reserve-manifest.json",
                "81d002e05becefe6746904fdb247b2a7825b644bf1ef81997609d2fb56ead503",
                "fca3e80f24f4fde0246ec28c436e7ba52e3266f53158a8adc628f8ac852019b9"),
    "deepcrack": ("deepcrack-confirmation.json",
                  "ce7ab064d63b554e222d4d01e0bc485b1d8dde6e4e5942e42647c773fdb677a8",
                  "b7e20eb3a21e865867319338278af1a3d879b730751cc6378485a53506fff73b"),
}
EXPERIMENTS_LF_SHA256 = "69534224482ef9cd92eaff63c48dbb71b657bf92075a3615c6614806e9586b75"


def digest(payload, normalize=False):
    return hashlib.sha256(payload.replace(b"\r\n", b"\n") if normalize else payload).hexdigest()


def verify_manifest(art, stage, frozen):
    if stage == "ratio-reserve":
        name = "ratio-reserve-manifest.json"
        if frozen.get("reserve_manifest") != name:
            raise ValueError("Freeze does not select the ratio reserve")
        payload = (art / name).read_bytes()
        valid = (digest(payload) == frozen["raw_reserve_manifest_sha256"] or
                 digest(payload, True) == frozen["lf_reserve_manifest_sha256"])
    else:
        name, raw, canonical = MANIFEST_HASHES[stage]
        field = "reserve_manifest_sha256" if stage == "reserve" else "deepcrack_manifest_sha256"
        if frozen.get(field) != raw:
            raise ValueError("Manifest freeze differs from the audited historical record")
        payload = (art / name).read_bytes()
        valid = digest(payload) == raw or digest(payload, True) == canonical
    if not valid:
        raise ValueError("Frozen case-selection manifest changed")
    return json.loads(payload)


def verify_sources(source_dir, frozen):
    canonical = "lf_source_sha256" in frozen
    expected = frozen["lf_source_sha256"] if canonical else frozen["raw_source_sha256"]
    for name, value in expected.items():
        if digest((source_dir / name).read_bytes(), canonical) != value:
            raise ValueError(f"Frozen implementation changed: {name}")
    # The early local freezes omitted this imported helper. Pin the dependency
    # separately to the immutable d155 implementation; this is a later audit.
    if digest((source_dir / "experiments.py").read_bytes(), True) != EXPERIMENTS_LF_SHA256:
        raise ValueError("Frozen reconstruction helper changed")
