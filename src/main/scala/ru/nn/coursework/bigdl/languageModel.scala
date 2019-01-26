package ru.nn.coursework.bigdl

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.text.{LabeledSentenceToSample, _}
import com.intel.analytics.bigdl.dataset.{DataSet, SampleToMiniBatch}
import com.intel.analytics.bigdl.example.languagemodel.PTBModel
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.example.languagemodel.Utils._
import com.intel.analytics.bigdl.models.rnn.SequencePreprocess
object languageModel {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.example").setLevel(Level.INFO)
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Train ptbModel on text")
        .setMaster("local[*]")
        .set("spark.task.maxFailures", "1")
        .set("spark.executor.memory", "6g")
        .set("spark.driver.memory", "6g")
        .set("spark.executor.cores", "4")
      val sc = new SparkContext(conf)
      Engine.init

      val trainDataFolder = "C:\\GRIAT\\NeuralNetworks\\CourseWork\\BigDLCourseWork\\src\\resources\\ptb\\simple-examples\\data"

      val (trainData, validData, testData, dictionary) = SequencePreprocess(
        trainDataFolder, param.vocabSize)

      val trainSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(trainData, param.numSteps)))
        .transform(TextToLabeledSentence[Float](param.numSteps))
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = None,
          fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      println("Size: " + trainSet.size())

      val validationSet = DataSet.rdd(sc.parallelize(
        SequencePreprocess.reader(validData, param.numSteps)))
        .transform(TextToLabeledSentence[Float](param.numSteps))
        .transform(LabeledSentenceToSample[Float](
          oneHot = false,
          fixDataLength = None,
          fixLabelLength = None))
        .transform(SampleToMiniBatch[Float](param.batchSize))

      val model = if (param.modelSnapshot.isDefined) {
        Module.loadModule[Float](param.modelSnapshot.get)
      } else {
        val curModel = PTBModel(
          inputSize = param.vocabSize,
          hiddenSize = param.hiddenSize,
          outputSize = param.vocabSize,
          numLayers = param.numLayers,
          keepProb = param.keepProb)
        curModel.reset()
        curModel
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new Adagrad[Float](learningRate = param.learningRate,
          learningRateDecay = param.learningRateDecay)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = false, dimension = 1)
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
          TimeDistributedCriterion[Float](
            CrossEntropyCriterion[Float](),
            sizeAverage = false, dimension = 1))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .optimize()
      sc.stop()
    })
  }
}
