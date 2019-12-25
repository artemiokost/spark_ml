package io.depa.predict

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Iris {

  def main(args: Array[String]) {

    val sc = SparkContext.getOrCreate(new SparkConf().setAppName("Main").setMaster("local[2]"))

    val data: RDD[String] = sc.textFile("data/mllib/iris_dataset.csv")

    val inputData = data.map(_.split(","))
      .map(arr => arr.slice(0, arr.length - 1))
      .map(for (e <- _) yield e.toDouble)
      .map(Vectors.dense)

    val labelMap = Map("setosa" -> 0, "versicolor" -> 1, "virginica" -> 2)

    val labeledData = data.map(_.split(",")).map(arr => {
      val label = arr(arr.length - 1)
      val vector = Vectors.dense(for (e <- arr.slice(0, arr.length - 1)) yield e.toDouble)
      LabeledPoint(labelMap(label), vector)
    })

    val summary = Statistics.colStats(inputData)
    println("Summary Mean:")
    println(summary.mean)
    println("Summary Variance:")
    println(summary.variance)
    println("Summary Non-zero:")
    println(summary.numNonzeros)

    val correlation = Statistics.corr(inputData, "pearson")
    println("Correlation Matrix:")
    println(correlation.toString)

    val splits = labeledData.randomSplit(Array(0.8, 0.2), 11L)
    val trainingData = splits(0)
    val testData = splits(1)

    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(labelMap.size)
      .run(trainingData)

    val predictionAndLabels = testData.map(p => (model.predict(p.features), p.label))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println("Model Accuracy on Test Data: " + accuracy)

    // Save and evaluate on random data
    model.save(sc, "model/logistic-regression")
    
    val nativeModel = LogisticRegressionModel.load(sc, "model/logistic-regression")
    val newData = Vectors.dense(Array[Double](1, 1, 1, 1))
    val prediction = nativeModel.predict(newData)
    println("Model Prediction on New Data = " + prediction)
  }
}
