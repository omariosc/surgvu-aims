#!/bin/bash
# Exit when the newest AIMS-run-B image reaches a TERMINAL import_status.
# Emits the status so a silent success and a silent failure look different.
exec > /users/sc20osc/joblogs/cat2_import_wait.log 2>&1
for i in $(seq 1 90); do
  OUT=$(/scratch/sc20osc/miccai-2026/gc_env/bin/python - <<'PY'
import json, ssl, urllib.request, warnings; warnings.filterwarnings("ignore")
TOK=open("/users/sc20osc/.gc_api_token").read().strip()
CTX=ssl.create_default_context(); CTX.check_hostname=False; CTX.verify_mode=ssl.CERT_NONE
r=urllib.request.Request("https://grand-challenge.org/api/v1/algorithms/images/?algorithm=4521d37a-c5ca-44c9-b225-fb8a9a0cc946",
  headers={"User-Agent":"Mozilla/5.0","Accept":"application/json","Authorization":"Bearer "+TOK})
res=json.loads(urllib.request.urlopen(r,context=CTX,timeout=90).read().decode()).get("results",[])
n=sorted(res,key=lambda x:str(x.get("created")))[-1]        # API is OLDEST-FIRST
print("%s %s" % (str(n.get("pk"))[:14], n.get("import_status")))
PY
)
  echo "$(date '+%H:%M:%S') $OUT"
  ST=$(echo "$OUT" | awk '{print tolower($2)}')
  case "$ST" in
    initialized|queued|started|re-queued|"") sleep 20 ;;
    *) echo "TERMINAL: $OUT"; exit 0 ;;
  esac
done
echo "TIMED OUT still non-terminal"; exit 1
