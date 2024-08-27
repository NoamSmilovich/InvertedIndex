#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=2
#SBATCH -o logs/log-%j.out

## --------------------------------------
## 1. Deploy a Spark cluster and submit a job
## --------------------------------------
export SPARK_HOME=/opt/ohpc/pub/libs/spark
$SPARK_HOME/deploy-spark-cluster.sh $@

