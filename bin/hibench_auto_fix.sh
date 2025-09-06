#!/usr/bin/env bash
set -euo pipefail

echo "[hibench-fix] start"

# --- sanity: контейнер и пути ---
if ! docker ps --format '{{.Names}}' | grep -q '^hibench$'; then
  echo "[hibench-fix] ERROR: контейнер 'hibench' не найден" >&2
  exit 1
fi

# 0) пути, которые нам нужны
LCFG=/opt/hibench/bin/functions/load_config.py
WFNS=/opt/hibench/bin/functions/workload_functions.sh
HCONF=/opt/hibench/conf/hibench.conf
SCONF=/opt/hibench/conf/spark.conf

# 1) Зальём внутрь контейнера шим load_config.py (надёжный, тихий по stdout)
echo "[hibench-fix] write load_config shim..."
docker exec -i hibench bash -lc "set -euo pipefail
cat > '$LCFG' <<'PY'
#!/usr/bin/env python3
import sys, os
from pathlib import Path

def log(*x):
    try: sys.stderr.write(' '.join(str(v) for v in x) + '\n')
    except Exception: sys.stderr.write(str(x) + '\n')

def main():
    if len(sys.argv) < 4:
        log('usage: load_config.py <conf_root> <workload_conf> <workload_folder> <patching_config>')
        sys.exit(2)

    # HiBench ждёт, что скрипт напечатает путь к env-файлу.
    _, conf_root, workload_conf, workload_folder, *rest = sys.argv

    env_dir  = Path('/opt/hibench/report') / workload_folder / 'conf'
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / ('micro.conf' if 'micro/' in workload_folder else 'workload.conf')

    if not env_file.exists():
        hdfs_master   = os.environ.get('HIBENCH_HDFS_MASTER', 'hdfs://namenode:8020')
        spark_master  = os.environ.get('SPARK_MASTER_URL',  'spark://spark-master:7077')
        spark_home    = os.environ.get('SPARK_HOME',        '/opt/spark')
        hadoop_home   = os.environ.get('HADOOP_HOME',       '/opt/hadoop')
        tests_jar     = '/opt/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-jobclient-3.3.6-tests.jar'
        input_hdfs    = '/HiBench/Wordcount/Input'
        output_hdfs   = '/HiBench/Wordcount/Output'

        with env_file.open('w') as f:
            f.write('# Auto-generated minimal env for HiBench micro/wordcount\n')
            f.write(f'HDFS_MASTER=\"{hdfs_master}\"\n')
            f.write(f'SPARK_HOME=\"{spark_home}\"\n')
            f.write(f'SPARK_MASTER=\"{spark_master}\"\n')
            f.write(f'HADOOP_HOME=\"{hadoop_home}\"\n')
            f.write(f'HADOOP_EXAMPLES_TEST_JAR=\"{tests_jar}\"\n')
            f.write(f'INPUT_HDFS=\"{input_hdfs}\"\n')
            f.write(f'OUTPUT_HDFS=\"{output_hdfs}\"\n')
            f.write('DATA_SCALE=\"tiny\"\n')
            f.write('NUM_PARTITIONS=4\n')

    print(str(env_file))

if __name__ == '__main__':
    main()
PY
chmod +x '$LCFG'
"

# 2) Уберём «болтовню» из workload_functions.sh (echo patching args -> stderr)
echo "[hibench-fix] silence workload_functions.sh echo..."
docker exec -i hibench bash -lc "set -euo pipefail
if [ -f '$WFNS' ]; then
  sed -i -E \"s/^(\\s*echo\\s+patching args=.*)([^>]|$)/\\1 1\\>\\&2/\" '$WFNS'
fi
"

# 3) Пропишем обязательные ключи в конфиги HiBench
echo "[hibench-fix] ensure hibench.conf/spark.conf keys..."
docker exec -i hibench bash -lc "set -euo pipefail
HCONF='$HCONF'; SCONF='$SCONF'
ensure(){ k=\"\$1\"; v=\"\$2\"; f=\"\$3\";
  mkdir -p \"\$(dirname \"\$f\")\"
  if grep -Eq \"^\\\${k}[[:space:]]\" \"\$f\"; then
    sed -i \"s|^\\\${k}[[:space:]].*|\\\${k}\t\\\${v}|\" \"\$f\"
  else
    printf \"%s\t%s\n\" \"\$k\" \"\$v\" >> \"\$f\"
  fi
}
ensure hibench.hadoop.home            /opt/hadoop                   \"\$HCONF\"
ensure hibench.hadoop.executable      /opt/hadoop/bin/hadoop        \"\$HCONF\"
ensure hibench.hadoop.configure.dir   /opt/hadoop/etc/hadoop        \"\$HCONF\"
ensure hibench.hdfs.master            hdfs://namenode:8020          \"\$HCONF\"
ensure hibench.configure.dir          /opt/hibench/conf             \"\$HCONF\"
ensure hibench.masters.hostnames      spark-master                  \"\$HCONF\"
ensure hibench.slaves.hostnames       spark-worker-1,spark-worker-2 \"\$HCONF\"
ensure hibench.report.dir             /opt/hibench/report           \"\$HCONF\"
ensure hibench.report.formats         \"plain,markdown,csv\"         \"\$HCONF\"
ensure spark.home                     /opt/spark                    \"\$SCONF\"
"

# 4) Подтянем sparkbench-jar, если отсутствует (тихо, один раз)
echo "[hibench-fix] ensure sparkbench jar (build if missing)..."
docker exec -i hibench bash -lc "set -euo pipefail
HCONF='$HCONF'
JAR_REL='assembly/target/sparkbench-assembly-8.0-SNAPSHOT-dist.jar'
JAR_ABS=\"/opt/hibench/sparkbench/\$JAR_REL\"
if [ ! -f \"\$JAR_ABS\" ]; then
  if command -v mvn >/dev/null 2>&1; then
    echo \"[hibench-fix] building sparkbench jar (one-time)...\"
    cd /opt/hibench/sparkbench
    mvn -q -DskipTests package
  else
    echo \"[hibench-fix] WARNING: mvn not found, skipping build — некоторые задачи могут не запуститься\" 1>&2
  fi
fi
if [ -f \"\$JAR_ABS\" ]; then
  if grep -q \"^hibench.sparkbench.jar\" \"\$HCONF\"; then
    sed -i \"s|^hibench.sparkbench.jar.*|hibench.sparkbench.jar\t\$JAR_REL|\" \"\$HCONF\"
  else
    printf \"hibench.sparkbench.jar\t%s\n\" \"\$JAR_REL\" >> \"\$HCONF\"
  fi
fi
"

echo "[hibench-fix] done"
