package ru.nn.coursework.bigdl.rnn

import com.intel.analytics.bigdl.dataset.text.{Dictionary, LabeledSentenceToSample, TextToLabeledSentence}
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, PaddingParam, SampleToMiniBatch}
import com.intel.analytics.bigdl.models.rnn.Train.{getClass, logger}
import com.intel.analytics.bigdl.models.rnn._
import com.intel.analytics.bigdl.nn.{CrossEntropyCriterion, Module, TimeDistributedCriterion}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object RNN {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  import Utils._
  val logger = Logger.getLogger(getClass)
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {

      val conf = Engine.createSparkConf()
        .setAppName("Train rnn on text")
        .set("spark.task.maxFailures", "1")
        .setMaster("local[*]")
      val sc = new SparkContext(conf)
      Engine.init

      val tokens = SequencePreprocess(
        param.dataFolder + "/train.txt",
        sc = sc,
        param.sentFile,
        param.tokenFile)

      val dictionary = Dictionary(tokens, param.vocabSize)
      dictionary.save(param.saveFolder)

      val maxTrainLength = tokens.map(x => x.length).max

      val valtokens = SequencePreprocess(
        param.dataFolder + "/val.txt",
        sc = sc,
        param.sentFile,
        param.tokenFile)
      val maxValLength = valtokens.map(x => x.length).max

      logger.info(s"maxTrain length = ${maxTrainLength}, maxVal = ${maxValLength}")

      val totalVocabLength = dictionary.getVocabSize() + 1
      val startIdx = dictionary.getIndex(SentenceToken.start)
      val endIdx = dictionary.getIndex(SentenceToken.end)
      val padFeature = Tensor[Float]().resize(totalVocabLength)
      padFeature.setValue(endIdx + 1, 1.0f)
      val padLabel = Tensor[Float](T(startIdx.toFloat + 1.0f))
      val featurePadding = PaddingParam(Some(Array(padFeature)),
        FixedLength(Array(maxTrainLength)))
      val labelPadding = PaddingParam(Some(Array(padLabel)),
        FixedLength(Array(maxTrainLength)))

      val trainSet = DataSet.rdd(tokens)
        .transform(TextToLabeledSentence[Float](dictionary))
        .transform(LabeledSentenceToSample[Float](totalVocabLength))
        .transform(SampleToMiniBatch[Float](
          param.batchSize,
          Some(featurePadding),
          Some(labelPadding)))

      val validationSet = DataSet.rdd(valtokens)
        .transform(TextToLabeledSentence[Float](dictionary))
        .transform(LabeledSentenceToSample[Float](totalVocabLength))
        .transform(SampleToMiniBatch[Float](param.batchSize,
          Some(featurePadding), Some(labelPadding)))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        val curModel = SimpleRNN(
          inputSize = totalVocabLength,
          hiddenSize = param.hiddenSize,
          outputSize = totalVocabLength)
        curModel.reset()
        curModel
      }

      val optimMethod = if (param.stateSnapshot.isDefined) {
        OptimMethod.load[Float](param.stateSnapshot.get)
      } else {
        new SGD[Float](learningRate = param.learningRate, learningRateDecay = 0.0,
          weightDecay = param.weightDecay, momentum = param.momentum, dampening = param.dampening)
      }

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = TimeDistributedCriterion[Float](
          CrossEntropyCriterion[Float](), sizeAverage = true)
      )

      if (param.checkpoint.isDefined) {
        optimizer.setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
      }

      if(param.overWriteCheckpoint) {
        optimizer.overWriteCheckpoint()
      }

      optimizer
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Loss[Float](
          TimeDistributedCriterion[Float](CrossEntropyCriterion[Float](), sizeAverage = true))))
        .setOptimMethod(optimMethod)
        .setEndWhen(Trigger.maxEpoch(param.nEpochs))
        .setCheckpoint(param.checkpoint.get, Trigger.everyEpoch)
        .optimize()
      sc.stop()
    })
  }
}
