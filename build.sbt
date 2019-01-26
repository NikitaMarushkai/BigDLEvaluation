name := "BigDLCourseWork"

version := "0.1"

scalaVersion := "2.11.12"

// https://mvnrepository.com/artifact/org.apache.spark/spark-core
libraryDependencies += "org.apache.spark" %% "spark-core" % "2.3.2"

// https://mvnrepository.com/artifact/org.apache.spark/spark-sql
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.2"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.3.2"

// https://mvnrepository.com/artifact/com.intel.analytics.bigdl.core/parent
libraryDependencies += "com.intel.analytics.bigdl.core" % "parent" % "0.7.2" pomOnly()

// https://mvnrepository.com/artifact/com.intel.analytics.bigdl/bigdl-SPARK
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.3" % "0.7.2"
