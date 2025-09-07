#!/usr/bin/env bash
set -euo pipefail

echo "[hibench-fix] start"

# --- sanity: контейнер и пути ---
if ! docker ps --format '{{.Names}}' | grep -q '^hibench$'; then
  echo "[hibench-fix] ERROR: контейнер 'hibench' не найден" >&2
  exit 1
fi

# Пути внутри контейнера
LCFG=/opt/hibench/bin/functions/load_config.py
WFNS=/opt/hibench/bin/functions/workload_functions.sh
EXLOG=/opt/hibench/bin/functions/execute_with_log.py
TSIZE=/opt/hibench/bin/functions/terminalsize.py
HCONF=/opt/hibench/conf/hibench.conf
SCONF=/opt/hibench/conf/spark.conf

# 1) Шим для load_config (стабильное чтение конфигов)
docker exec -i hibench bash -lc "cat > $LCFG" <<'PY'
#!/usr/bin/env python3
import os

def get_conf(key, default=''):
    # 1) окружение
    if key in os.environ:
        return os.environ[key]
    # 2) conf/hibench.conf (tab или пробел как разделитель)
    conf_path = '/opt/hibench/conf/hibench.conf'
    if os.path.isfile(conf_path):
        with open(conf_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2 and parts[0] == key:
                    return ' '.join(parts[1:])
    return default
PY
docker exec -it hibench bash -lc "chmod +x $LCFG"

# 2) Тише логи, чтобы не ломать парсинг
docker exec -i hibench bash -lc "sed -i 's/^echo_log()/#echo_log()/g' $WFNS || true"

# 3) Перевод helper-скриптов на Python 3
#    - меняем шебанги на python3
#    - правим print "..." -> print("...")
docker exec -i hibench bash -lc "sed -i '1s|/usr/bin/env python2|/usr/bin/env python3|; 1s|/usr/bin/env python|/usr/bin/env python3|' $EXLOG $TSIZE || true"
docker exec -i hibench bash -lc "python3 - <<'PY'
from pathlib import Path
import re
files = ['/opt/hibench/bin/functions/terminalsize.py', '/opt/hibench/bin/functions/execute_with_log.py']
for p in files:
    fp = Path(p)
    if not fp.exists(): 
        continue
    s = fp.read_text(encoding='utf-8', errors='ignore')
    # Простейшая замена print "..." -> print("...")
    s = re.sub(r'(?m)^[ \t]*print (.+)$', lambda m: 'print(' + m.group(1).rstrip() + ')', s)
    fp.write_text(s, encoding='utf-8')
print('patched python3 prints')
PY"

# 4) Гарантированные ключи в конфиге HiBench
docker exec -i hibench bash -lc "grep -q '^hibench.hadoop.home' $HCONF || echo -e 'hibench.hadoop.home\t/opt/hadoop' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^hibench.hadoop.executable' $HCONF || echo -e 'hibench.hadoop.executable\t/opt/hadoop/bin/hadoop' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^hibench.hadoop.configure.dir' $HCONF || echo -e 'hibench.hadoop.configure.dir\t/opt/hadoop/etc/hadoop' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^hibench.hdfs.master' $HCONF || echo -e 'hibench.hdfs.master\thdfs://namenode:8020' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^hibench.spark.home' $HCONF || echo -e 'hibench.spark.home\t/opt/spark' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^spark.master' $HCONF || echo -e 'spark.master\tspark://spark-master:7077' >> $HCONF"
docker exec -i hibench bash -lc "grep -q '^hibench.scale.profile' $HCONF || echo -e 'hibench.scale.profile\ttiny' >> $HCONF"

# 5) Минимальные настройки Spark (если нет)
docker exec -i hibench bash -lc "grep -q '^spark.executor.memory' $SCONF || echo -e 'spark.executor.memory\t2g' >> $SCONF"
docker exec -i hibench bash -lc "grep -q '^spark.executor.cores'  $SCONF || echo -e 'spark.executor.cores\t1' >> $SCONF"
docker exec -i hibench bash -lc "grep -q '^spark.driver.memory'   $SCONF || echo -e 'spark.driver.memory\t1g' >> $SCONF"

# 6) Найти sparkbench jar и прописать путь
docker exec -i hibench bash -lc '
JAR=$(ls -1 /opt/hibench/sparkbench/**/target/*assembly*.jar 2>/dev/null | head -n1 || true)
if [ -z "$JAR" ]; then
  if command -v mvn >/dev/null 2>&1; then
    cd /opt/hibench
    export JAVA_HOME=${JAVA_HOME:-/opt/java8}; export PATH="$JAVA_HOME/bin:$PATH"
    mvn -q -DskipTests -pl common,sparkbench -am package || true
    JAR=$(ls -1 /opt/hibench/sparkbench/**/target/*assembly*.jar 2>/dev/null | head -n1 || true)
  fi
fi
if [ -n "$JAR" ]; then
  if grep -q "^hibench.sparkbench.jar" '"$HCONF"'; then
    sed -i "s|^hibench.sparkbench.jar.*|hibench.sparkbench.jar\t$JAR|" '"$HCONF"'
  else
    printf "hibench.sparkbench.jar\t%s\n" "$JAR" >> '"$HCONF"'
  fi
fi
'

echo "[hibench-fix] done"
