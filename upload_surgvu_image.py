#!/usr/bin/env python
"""Upload a SurgVU algorithm container to grand-challenge.org, end to end.

    gc_env/bin/python upload_surgvu_image.py --target cat2 --file <artifact.tar.gz> [--yes]

A SurgVU-specific sibling of ORENA's upload_algorithm_image.py. That file is a VALIDATED
INSTRUMENT and is deliberately NOT modified; every safeguard below is copied from it, and
each was earned by a real failure in this campaign:
  * TARGET MATCHED BY PK, NEVER BY TITLE (title-matching lost the 2026-08-04 submission).
  * sha256 verified against an expected digest BEFORE anything is sent.
  * RepoTag read from INSIDE the archive (survives a rename) and must name the target.
    ★ This matters acutely here: surgvu26-cat1-aims_*.tar.gz lives in the SAME directory,
      so a careless --file would upload the Cat-1 detector into the Cat-2 algorithm.
  * `creator` is read FROM THE FORM, never guessed — posting a username makes Django
    return HTTP 200 and create NOTHING, silently (cost two ORENA uploads in Aug 2026).
  * import_status is polled afterwards; all three past ORENA failures were visible only there.
  * It uploads and attaches. It does NOT submit to a phase - that stays a separate decision.
"""
import argparse, hashlib, json, os, re, sys, tarfile, time, warnings
warnings.filterwarnings("ignore")
import requests, gcapi
import urllib3; urllib3.disable_warnings()

BASE = "https://grand-challenge.org"
TOKEN_FILE = "/users/sc20osc/.gc_api_token"
PW_FILE = "/users/sc20osc/.gc_pw"
USERNAME = "omarchoudhry"

# (slug, pk, word that MUST appear in the archive's RepoTag)
TARGETS = {
    "cat2": ("aims-surgvu26-cat-2-deterministic-answerer",
             "4521d37a-c5ca-44c9-b225-fb8a9a0cc946", "cat2"),
    "cat1": ("aims-surgvu26-cat-1-tool-detector",
             "10d3a54a-7974-425f-a2c2-796762decfc4", "cat1"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def repo_tag(path):
    try:
        with tarfile.open(path, "r:gz") as t:
            m = t.extractfile("manifest.json")
            return (json.load(m)[0].get("RepoTags") or ["<none>"])[0]
    except Exception as e:
        return "<unreadable: %s>" % type(e).__name__


def session_login():
    s = requests.Session(); s.verify = False
    s.headers["User-Agent"] = "Mozilla/5.0"
    L = BASE + "/accounts/login/"
    r = s.get(L, timeout=60)
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    pw = open(PW_FILE).read().strip()
    r = s.post(L, data={"login": USERNAME, "password": pw, "csrfmiddlewaretoken": csrf},
               headers={"Referer": L}, timeout=120)
    if "Sign out" not in r.text and "/users/" not in r.url:
        sys.exit("FATAL: session login failed")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--file", required=True)
    ap.add_argument("--expect-sha256", default=None)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    slug, pk, word = TARGETS[a.target]
    if not os.path.isfile(a.file):
        sys.exit("FATAL: no such file: " + a.file)
    size = os.path.getsize(a.file)
    print("  artifact : %s" % a.file)
    print("  bytes    : %d (%.2f GB)" % (size, size / 1073741824))
    digest = sha256(a.file)
    print("  sha256   : %s" % digest)
    if a.expect_sha256 and digest != a.expect_sha256:
        sys.exit("FATAL: sha256 MISMATCH - expected %s" % a.expect_sha256)
    tag = repo_tag(a.file)
    print("  RepoTag  : %s" % tag)
    if word not in tag.lower().replace("-", ""):
        sys.exit("FATAL: RepoTag %r does not name target %r (word %r). Refusing - this is "
                 "the wrong-artifact swap class." % (tag, a.target, word))
    print("  target   : %s  (pk %s)" % (slug, pk))
    if not a.yes and input("  proceed? [yes/NO] ").strip().lower() != "yes":
        sys.exit("aborted")

    c = gcapi.Client(token=open(TOKEN_FILE).read().strip(), base_url=BASE + "/api/v1/")
    print("  [1/3] uploading bytes via the API ...")
    t0 = time.time()
    with open(a.file, "rb") as fh:
        up = c.uploads.upload_fileobj(fileobj=fh, filename=os.path.basename(a.file))
    up_pk = up["pk"] if isinstance(up, dict) else up.pk
    print("        done in %.1f min -> user_upload %s" % ((time.time() - t0) / 60, up_pk))

    print("  [2/3] attaching to the algorithm (web form) ...")
    s = session_login()
    F = "%s/algorithms/%s/images/create/" % (BASE, slug)
    r = s.get(F, timeout=60)
    csrf = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    def field(name, default=""):
        m = re.search(r'name="%s"[^>]*value="([^"]*)"' % name, r.text)
        return m.group(1) if m else default
    creator = field("creator")
    if not creator:
        sys.exit("FATAL: could not read `creator` from the form; refusing to post a "
                 "request that would 200 and create nothing")
    print("        creator field from form: %r" % creator)
    r = s.post(F, data={"csrfmiddlewaretoken": csrf, "algorithm": pk,
                        "creator": creator, "user_upload": up_pk},
               headers={"Referer": F}, timeout=300)
    print("        POST -> %s  %s" % (r.status_code, r.url[:90]))
    if r.url.rstrip("/").endswith("create"):
        print("        ⚠ redirected BACK to the form -> the attach FAILED")

    print("  [3/3] polling import_status ...")
    for i in range(60):
        time.sleep(20)
        # gcapi's c("path") form is read as c(method,...) -> h11 "Illegal method
        # characters". Poll the REST API directly with the bearer token instead.
        imgs = requests.get(BASE + "/api/v1/algorithms/images/?limit=20",
                            headers={"Authorization": "BEARER " + open(TOKEN_FILE).read().strip()},
                            timeout=60).json()
        mine = [im for im in imgs.get("results", []) if pk in str(im.get("algorithm", ""))]
        if mine:
            newest = sorted(mine, key=lambda x: x.get("created", ""))[-1]   # OLDEST-FIRST api
            st = newest.get("import_status")
            print("        %s  image %s  created=%s import_status=%s"
                  % (time.strftime("%H:%M:%S"), str(newest.get("pk", "?"))[:8],
                     str(newest.get("created"))[:19], st))
            if st and st.lower() not in ("initialized", "queued", "started", "re-queued"):
                print("\n  RESULT: import_status = %s" % st)
                return 0 if str(st).lower() == "completed" else 1
    print("  timed out waiting for import; check the algorithm page.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
