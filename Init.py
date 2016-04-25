__author__ = 'Andrey Mironoff'
import os
import sys
import numpy
# Path for spark source folder
os.environ['SPARK_HOME']="/usr/local/bin/spark-1.3.1-bin-hadoop2.6"

# Append pyspark  to Python Path
sys.path.append("/usr/local/bin/spark-1.3.1-bin-hadoop2.6/python/")
sys.path.append("/usr/local/bin/spark-1.3.1-bin-hadoop2.6/python/lib/py4j-0.8.2.1-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf


    print ("Yay, Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)
