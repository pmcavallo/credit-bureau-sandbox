@echo off
set JAVA_HOME=C:\Progra~1\Eclipse~1\jdk-8.0.452.9-hotspot
set SPARK_HOME=C:\spark\spark-3.5.6-bin-hadoop3
call conda activate spark310
python spark_test.py
