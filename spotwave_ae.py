from __future__ import annotations
import logging
import json
import numpy as np
from dataclasses import asdict, dataclass
from waveline import SpotWave
from waveline.spotwave import AERecord, TRRecord
import sys

logging.basicConfig(level=logging.INFO)

@dataclass
class HitRecord(AERecord):
    """All fields from AERecord + fields for transient data."""
    samples: int
    data: np.ndarray

def merge_ae_tr_records(generator):
    dict_ae: dict[int, AERecord] = {}
    dict_tr: dict[int, TRRecord] = {}

    for record in generator:
        if isinstance(record, AERecord):
            dict_ae[record.trai] = record
        elif isinstance(record, TRRecord):
            dict_tr[record.trai] = record

        trais_match = set(dict_ae.keys()).intersection(dict_tr.keys())
        for trai in trais_match:
            ae_record = dict_ae.pop(trai)
            tr_record = dict_tr.pop(trai)
            yield HitRecord(
                **asdict(ae_record),
                samples=tr_record.samples,
                data=tr_record.data,
            )

def main():
    try:
        ports = SpotWave.discover()
        if not ports:
            print("[Error] No SpotWave device found.", file=sys.stderr, flush=True)
            return

        port = ports[0]
        with SpotWave(port) as sw:
            print("[Connected] SpotWave device opened.", file=sys.stderr, flush=True)
            print(sw.get_info(), file=sys.stderr, flush=True)

            sw.set_continuous_mode(False)
            sw.set_ddt(10_000)
            sw.set_status_interval(2)
            sw.set_threshold(1000)
            sw.set_tr_enabled(True)
            sw.set_tr_pretrigger(200)
            sw.set_tr_postduration(0)
            sw.set_filter(100e3, 450e3, 4)

            print("[Setup Complete] Waiting for AE hits...", file=sys.stderr, flush=True)

            for record in merge_ae_tr_records(sw.acquire()):
                if isinstance(record, HitRecord):
                    record_dict = asdict(record)
                    record_dict["data"] = record.data.tolist()
                    print(json.dumps(record_dict), flush=True)

    except Exception as e:
        print(f"[Exception] SpotWave acquisition error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
