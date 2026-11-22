#!/bin/bash
# Build recipe of this solver, as used for every build of the paper's runs.
# Run it from this directory:  bash build.sh <output directory>   ->  <output directory>/solver
set -e
set -o pipefail
CC=${CC:-gcc}
command -v "$CC" > /dev/null || { echo "no C compiler ($CC) on PATH"; exit 1; }
[ -f src/main.c ] || { echo "src/main.c is missing"; exit 1; }
touch params.txt
python3 - <<'PY' || exit 1
import re, sys
t = open("params.txt").read().split()
ok = re.compile(r"^(-p \d{1,4}|-s \d+(\.\d+)?|-q \d{1,6})$")
bad = [" ".join(t[i:i + 2]) for i in range(0, len(t), 2) if not ok.match(" ".join(t[i:i + 2]))]
if bad or len(t) % 2:
    sys.exit("params.txt may only hold the options -p <int>, -s <seconds>, -q <int> (found: %s)" % (bad or t))
PY
SRCS=$(ls src/*.c | grep -v '^src/chils\.c$')
if [ "$2" = check ]; then                   # compile errors only, no binary
  "$CC" -std=gnu17 -O1 -I include -fopenmp -fsyntax-only $SRCS 2>&1 | tail -n 150
  exit 0
fi
"$CC" -std=gnu17 -O3 -march=native -I include -fopenmp -DNDEBUG $SRCS -o "$1/prog" 2>&1 | tail -n 150
cp params.txt "$1/params.txt"
cat > "$1/solver" <<'LAUNCHER'
#!/usr/bin/env python3
"""solver <instance> <solution> -t <seconds> -seed <int>: runs the host on ONE thread (its option -c 1, and the OpenMP
runtime limited to one thread) from a cold start; the tuning options of params.txt (-p, -s, -q) are passed on."""
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
a = sys.argv[1:]
inst, sol = a[0], a[1]
T = a[a.index("-t") + 1]
seed = a[a.index("-seed") + 1]
extra = open(os.path.join(here, "params.txt")).read().split()
env = dict(os.environ)
env.update(OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_DYNAMIC="false", OMP_PROC_BIND="false")
cmd = [os.path.join(here, "prog"), "-g", inst, "-o", sol, "-t", T, "-r", seed, "-c", "1"] + extra
os.execve(cmd[0], cmd, env)
LAUNCHER
chmod +x "$1/solver"
