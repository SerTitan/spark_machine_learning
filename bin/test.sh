#!/usr/bin/env bash
set -Eeuo pipefail

CN=hibench

# 1) Патч execute_with_log.py так, чтобы не было bytes→str ошибок
docker exec -i "$CN" bash -lc '
  set -Eeuo pipefail
  FILE=/opt/hibench/bin/functions/execute_with_log.py
  test -f "$FILE" || { echo "❌ Нет файла: $FILE"; exit 1; }
  cp -a "$FILE" "$FILE.bak.$(date +%s)"

  # Создаём патчер-файл атомарно и безопасно (quoted heredoc!)
  cat > /tmp/fix_execute_with_log.py <<'"'"'PY'"'"'
from pathlib import Path

fp = Path("/opt/hibench/bin/functions/execute_with_log.py")
lines = fp.read_text(encoding="utf-8").splitlines(True)

# 1) Вставим хелпер _to_text() сразу после блока import-ов, если его нет
txt = "".join(lines)
if "_to_text(" not in txt:
    out = []
    inserted = False
    i = 0
    n = len(lines)
    while i < n:
        out.append(lines[i])
        s = lines[i].lstrip()
        if not inserted and not (s.startswith("import ") or s.startswith("from ")):
            out.append("\n")
            out.append("def _to_text(x):\n")
            out.append("    return x.decode(\"utf-8\", \"ignore\") if isinstance(x, (bytes, bytearray)) else str(x)\n")
            out.append("\n")
            inserted = True
        i += 1
    if not inserted:
        out.append("\n")
        out.append("def _to_text(x):\n")
        out.append("    return x.decode(\"utf-8\", \"ignore\") if isinstance(x, (bytes, bytearray)) else str(x)\n")
        out.append("\n")
    lines = out

# 2) Нормализуем строки записи в лог для stdout/stderr
fixed = []
for l in lines:
    sl = l.strip()
    if "log_file.write(" in sl and "line" in sl and "_to_text(line)" not in sl:
        indent = l[:len(l) - len(l.lstrip(" "))]
        fixed.append(f"{indent}log_file.write(_to_text(line)+\"\\n\")\n")
    elif "log_file.write(" in sl and "errline" in sl and "_to_text(errline)" not in sl:
        indent = l[:len(l) - len(l.lstrip(" "))]
        fixed.append(f"{indent}log_file.write(_to_text(errline)+\"\\n\")\n")
    else:
        # На случай «разорванных» строк (незакрытых кавычек) – добьём корректной формой
        if "log_file.write(_to_text(line)+" in sl and "\\n" in sl and not sl.endswith('")'):
            indent = l[:len(l) - len(l.lstrip(" "))]
            fixed.append(f"{indent}log_file.write(_to_text(line)+\"\\n\")\n")
        elif "log_file.write(_to_text(errline)+" in sl and "\\n" in sl and not sl.endswith('")'):
            indent = l[:len(l) - len(l.lstrip(" "))]
            fixed.append(f"{indent}log_file.write(_to_text(errline)+\"\\n\")\n")
        else:
            fixed.append(l)

fp.write_text("".join(fixed), encoding="utf-8")
print("OK: normalized write() lines")
PY

  python3 /tmp/fix_execute_with_log.py
  python3 -m py_compile /opt/hibench/bin/functions/execute_with_log.py && echo "✅ Python синтаксис OK"
'

# 2) (Опционально) быстрый прогон wordcount, если нужно:
# docker exec -i "$CN" bash -lc '
#   set -Eeuo pipefail
#   HIBENCH_DIR=/opt/hibench
#   sed -i -E "s/^(hibench\.scale\.profile).*/\1 tiny/" "$HIBENCH_DIR/conf/hibench.conf"
#   echo "➡️ prepare (wordcount)"; "$HIBENCH_DIR/bin/workloads/micro/wordcount/prepare/prepare.sh"
#   echo "➡️ run (spark wordcount)"; "$HIBENCH_DIR/bin/workloads/micro/wordcount/spark/run.sh"
#   echo "➡️ report"; "$HIBENCH_DIR/bin/report.sh"
# '

echo "✔ done"
