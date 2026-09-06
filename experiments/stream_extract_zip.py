"""Stream-extract COMPLETE mp4 entries from a still-downloading SurgVU zip.

The zip uses deflate (method 8) with trailing data descriptors (flags bit 3),
so the compressed size is not in the local header. We locate each local file
header by its PK\\x03\\x04 signature, decompress the deflate stream that follows
incrementally, and stop at its natural end (zlib reports end-of-stream). An entry
is only written if its full deflate stream is present in the bytes downloaded so
far -- partially-downloaded trailing entries are skipped (extracted on a re-run).

Idempotent: skips files already extracted with the right size.
Usage:
  python stream_extract_zip.py --zip <...>.zip --out <dir> [--max_cases N] [--only_mp4]
"""
import argparse
import mmap
import os
import struct
import zlib

SIG_LFH = b"PK\x03\x04"
SIG_CD = b"PK\x01\x02"


def iter_local_headers(mm):
    i = 0
    n = len(mm)
    while True:
        j = mm.find(SIG_LFH, i)
        if j < 0 or j + 30 > n:
            return
        if mm[j:j + 4] == SIG_CD:  # reached central dir
            return
        flags = struct.unpack("<H", mm[j + 6:j + 8])[0]
        method = struct.unpack("<H", mm[j + 8:j + 10])[0]
        csize = struct.unpack("<I", mm[j + 18:j + 22])[0]
        nlen = struct.unpack("<H", mm[j + 26:j + 28])[0]
        elen = struct.unpack("<H", mm[j + 28:j + 30])[0]
        name = mm[j + 30:j + 30 + nlen].decode("utf-8", "replace")
        data_start = j + 30 + nlen + elen
        yield name, method, flags, csize, data_start
        i = data_start  # next search begins after this header's data start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only_mp4", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0, help="0 = unlimited")
    args = ap.parse_args()

    fh = open(args.zip, "rb")
    mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    total = len(mm)
    print(f"zip size now: {total/1e9:.2f} GB", flush=True)

    extracted, skipped, partial = 0, 0, 0
    cases = set()
    for name, method, flags, csize, ds in iter_local_headers(mm):
        if name.endswith("/"):
            continue
        if args.only_mp4 and not name.lower().endswith(".mp4"):
            continue
        if "__MACOSX" in name or os.path.basename(name).startswith("._"):
            continue
        case = name.split("/")[1] if name.count("/") >= 1 and name.startswith("surgvu24/") else ""
        if args.max_cases and case and case not in cases and len(cases) >= args.max_cases:
            continue
        out_path = os.path.join(args.out, name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        CHUNK = 16 << 20
        # decompress incrementally from the mmap (no giant slice); if the deflate
        # stream is truncated (still downloading) -> partial, leave for a re-run.
        try:
            if method == 0:  # stored
                if csize == 0 or ds + csize > total:
                    partial += 1
                    continue
                written = csize
                with open(tmp, "wb") as o:
                    o.write(mm[ds:ds + csize])
            else:  # deflate
                d = zlib.decompressobj(-zlib.MAX_WBITS)
                pos = ds
                written = 0
                with open(tmp, "wb") as o:
                    while pos < total and not d.eof:
                        chunk = mm[pos:min(pos + CHUNK, total)]
                        pos += len(chunk)
                        out = d.decompress(chunk)
                        if out:
                            o.write(out); written += len(out)
                    tail = d.flush()
                    if tail:
                        o.write(tail); written += len(tail)
                if not d.eof:
                    os.remove(tmp)
                    partial += 1  # stream not fully downloaded yet
                    continue
        except (zlib.error, Exception):
            if os.path.exists(tmp):
                os.remove(tmp)
            partial += 1
            continue
        if os.path.exists(out_path) and os.path.getsize(out_path) == written:
            os.remove(tmp)
            skipped += 1
            cases.add(case)
            continue
        os.replace(tmp, out_path)
        extracted += 1
        cases.add(case)
        print(f"  + {name}  ({written/1e6:.0f} MB)", flush=True)

    print(f"extracted={extracted} skipped(existing)={skipped} "
          f"partial(not-yet-downloaded)={partial} cases_with_files={len(cases)}")


if __name__ == "__main__":
    main()
