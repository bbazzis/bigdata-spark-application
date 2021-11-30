package upm.bd
import Array._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, when, concat, lit, unix_timestamp, udf}
import org.apache.spark.SparkContext._
import java.io.PrintWriter
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression.{LinearRegression, GeneralizedLinearRegression, RandomForestRegressionModel, RandomForestRegressor, GBTRegressor}
import org.apache.spark.ml.feature.UnivariateFeatureSelector
import org.apache.spark.ml.feature.VarianceThresholdSelector
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator

object MyApp {

	/*
	** These columns have been forbidden by the assignment rules
	** as these values are unknown at the time of flight departure
	*/
	val ForbiddenColumns = Array(
		"ArrTime",
		"ActualElapsedTime",
		"AirTime",
		"TaxiIn",
		"Diverted",
		"CarrierDelay",
		"WeatherDelay",
		"NASDelay",
		"SecurityDelay",
		"LateAircraftDelay"
	)

	/*
	** The following columns are exluded because
	** they were either mostly null or did not have an effect on the resulting delay
	*/
	val ExcludedColumns = Array(
		"Cancelled",
		"CancellationCode",
		"TailNum",
		"TaxiOut",
		"Year",
		"Month"
	)
		
	// Numerical Variables
	val IntColumns = Array(
		"DepTime",
		"CRSDepTime",
		"CRSArrTime",
		"FlightNum",
		"CRSElapsedTime",
		"DepDelay",
		"Distance",
		"TimeAsEpoch"
	)

	// Categorical variables
	val CatColumns = Array(
		"Origin",
		"Dest",
		"UniqueCarrier",
		"DayofMonth",
		"Season",
		"DayOfWeek"
	)

	val IndexedCatColumns = Array(
		"Ind_Origin",
		"Ind_Dest",
		"Ind_UniqueCarrier",
		"Ind_DayofMonth",
		"Ind_Season",
		"Ind_DayOfWeek"
	)

	val EncodedCatColumns = Array(
		"Enc_Origin",
		"Enc_Dest",
		"Enc_UniqueCarrier",
		"Enc_DayofMonth",
		"Enc_Season",
		"Enc_DayOfWeek"
	)

	val LabelColumn = "ArrDelay"

	def main(args : Array[String]) : Unit = {
		Logger.getLogger("org").setLevel(Level.WARN)
		
		if (args.length == 0) {
			println("Please enter the directory of the data file")
			System.exit(0)
		}

		val filename = args(0)
		println("filename = " + filename)
		// filename = "/tmp/csv/1987_min.csv"
		// TODO: Check whether the file exists at the location
		// TODO: Check if the file is of correct type
		// TODO: Check if file can be read without an issue
		
		//select algorithm
		if (args.length == 1)
			println("No machine learning algorithm has been selected, all of them will be executed sequentially.")

		val validAlgorithms = List("LR", "GLR", "RF", "GBR")
		var selectedAlgorithm = "NONE"
		if (args.length > 1) {
			val input = args(1).toUpperCase()
			if (!(validAlgorithms.contains(input))) {
				println("Please either select a valid algorithm or do not select anything.")
				println("Available algorithms: "+validAlgorithms)
				System.exit(0)
			} else {
				selectedAlgorithm = input
				println("Algorithm selected = " + selectedAlgorithm)
			}
		}

		val spark = org.apache.spark.sql.SparkSession
			.builder()
			.appName("Group 13 - Big Data Assignment - Flight Delay Predictor")
			.config("some option", "value") // TODO: Remove or fix
			.enableHiveSupport()
			.getOrCreate()
		
		// Skip malformed data
		// TODO: Probably need to read into RDD(s)
		var data = spark.read.option("header",true).option("mode", "DROPMALFORMED").csv(filename)
		
		// Transform cancelled field to int and then drop the cancelled flights
		data = data.withColumn("Cancelled", col("Cancelled").cast("integer"))
		data = data.filter("Cancelled == 0")

		//Transform Month variable into Season
		data = data.withColumn("Month", col("Month").cast("string"))
		data = data.withColumn("Season", when(col("Month") === 12 || col("Month") === 1 || col("Month") === 2,"Winter")
			.when(col("Month") === 3 || col("Month") === 4 || col("Month") === 5,"Spring")
			.when(col("Month") === 6 || col("Month") === 7 || col("Month") === 8,"Summer")
			.otherwise("Autumn"))

		// Using string columns as categorical data
		val indexer = new StringIndexer()
			.setInputCols(CatColumns)
			.setOutputCols(IndexedCatColumns)
		val indexed = indexer.fit(data).transform(data)

		val encoder = new OneHotEncoder()
			.setInputCols(IndexedCatColumns)
			.setOutputCols(EncodedCatColumns)
		data = encoder.fit(indexed).transform(indexed)
		
		// Merge date columns into a single timeAsEpoch Column
		val fixDateZeroPaddingUDF = udf((year:String, month:String, day:String) => {
			var paddedDate = "%04d".format(year.toInt) +
					 "-" +
					 "%02d".format(month.toInt) +
					 "-" + 
					 "%02d".format(day.toInt)
			paddedDate
		})
		data = data.withColumn("TimeAsEpoch", unix_timestamp(
									fixDateZeroPaddingUDF(col("Year"), col("Month"), col("DayOfMonth")), "yyyy-LL-dd"))
		
		// Drop all excluded columns	
		data = data.drop((ForbiddenColumns++ExcludedColumns): _*)

		// Cast int columns to int
		for (colName <- IntColumns++Array(LabelColumn))
			data = data.withColumn(colName, col(colName).cast("integer"))

		// TODO: remove the line below before the final submission
		data.printSchema()

		// --------------------------------------------------------------------------------------------------
		// Machine learning begins
		// --------------------------------------------------------------------------------------------------
		
		//SPLITING DATA
		val split = data.randomSplit(Array(0.8,0.2))
		val trainingData = split(0)
		val testData = split(1)

		val functions = Map(
			"LR" -> applyLinearRegressionModel _,
			"GLR" -> applyGeneralizedLinearRegressionModel _,
			"RF" -> applyRandomForestRegressionModel _,
			"GBR" -> applyGradientBoostedRegressionModel _
		)

		val numberOfFeatures = 14				// Between 1-14 variables
		var numberOfAlgorithmsToRun = 1			// 1 or 4 algorithms
		val numberOfEvaluationParameters = 2 	// 2 outputs: RMSE and R^2
		if (selectedAlgorithm == "NONE")
			numberOfAlgorithmsToRun = 4

		var univariateResult = Array.ofDim[Double](numberOfFeatures, numberOfAlgorithmsToRun, numberOfEvaluationParameters) 

		// Feature selection and comparison of rmse and r^2
		for(i <- 1 to univariateResult.size){
			println("Number of variables: " + i)

			val assembler = new VectorAssembler()
				.setInputCols(IntColumns++EncodedCatColumns)
				.setOutputCol("features")
				.setHandleInvalid("skip")
		
			// val output = assembler.transform(trainingData)
			val selector = new UnivariateFeatureSelector()
				.setFeatureType("continuous")
				.setLabelType("categorical")
				.setSelectionMode("numTopFeatures")
				.setSelectionThreshold(i)
				.setFeaturesCol("features")
				.setLabelCol(LabelColumn)
				.setOutputCol("selectedFeatures")

			// val result = selector.fit(output).transform(output)
			val normalizer = new Normalizer()
				.setInputCol("selectedFeatures")
				.setOutputCol("normFeatures")
				.setP(1.0)

			val pipeline = new Pipeline().setStages(Array(assembler, selector, normalizer))
			val pipelineModel = pipeline.fit(trainingData)
			val fssTrainingData = pipelineModel.transform(trainingData)
			val fssTestData = pipelineModel.transform(testData)

			if(selectedAlgorithm == "NONE") {
				univariateResult(i-1)(0) = applyLinearRegressionModel(fssTrainingData, fssTestData)
				univariateResult(i-1)(1) = applyGeneralizedLinearRegressionModel(fssTrainingData, fssTestData)
				univariateResult(i-1)(2) = applyRandomForestRegressionModel(fssTrainingData,fssTestData)
				univariateResult(i-1)(3) = applyGradientBoostedRegressionModel(fssTrainingData,fssTestData)
			} else {
				val algorithm = functions(selectedAlgorithm)
				univariateResult(i-1)(0) = algorithm(fssTrainingData,fssTestData);
			}
		}

		println("Number of variables - rmse | r2")
		for(i <- 0 to univariateResult.size - 1) {
			if(selectedAlgorithm == "NONE") {
				println((i+1) + " LR  - " + univariateResult(i)(0)(0) + " | " + univariateResult(i)(0)(1))
				println((i+1) + " GLR - " + univariateResult(i)(1)(0) + " | " + univariateResult(i)(1)(1))
				println((i+1) + " RF  - " + univariateResult(i)(2)(0) + " | " + univariateResult(i)(2)(1))
				println((i+1) + " GBR - " + univariateResult(i)(3)(0) + " | " + univariateResult(i)(3)(1))
			} else {
				println((i+1) + " " + selectedAlgorithm + " - " + univariateResult(i)(0)(0) + " | " + univariateResult(i)(0)(1))
			}
		}
	}

	def applyUnivariateFilter( data:DataFrame, a:Double) : DataFrame = {
		val assembler = new VectorAssembler()
			.setInputCols(IntColumns++EncodedCatColumns)
			.setOutputCol("features")
			.setHandleInvalid("skip")
		
		val output = assembler.transform(data)
		val selector = new UnivariateFeatureSelector()
			.setFeatureType("continuous")
			.setLabelType("categorical")
			.setSelectionMode("numTopFeatures")
			.setSelectionThreshold(a)
			.setFeaturesCol("features")
			.setLabelCol(LabelColumn)
			.setOutputCol("selectedFeatures")

		val result = selector.fit(output).transform(output)

		return result
   }

   def applyVarianceThresholdSelector( data:DataFrame ) : Unit = {
		val selector = new VarianceThresholdSelector()
			.setVarianceThreshold(2.0)
			.setFeaturesCol("features")
			.setOutputCol("selectedFeatures")

		val result = selector.fit(data).transform(data)

		println(s"Output: Features with variance lower than" +
		s" ${selector.getVarianceThreshold} are removed.")
		result.show()
   }

   def applyLinearRegressionModel(normTrainingData:DataFrame , normTestData:DataFrame) : Array[Double] = {
		val lr = new LinearRegression()
  			.setFeaturesCol("normFeatures")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setElasticNetParam(0.8)
		
		val lrModel = lr.fit(normTrainingData)
		val predictions = lrModel.transform(normTestData)
		val rmse_lr = new RegressionEvaluator()
			.setMetricName("rmse")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)
		
		val r2_lr = new RegressionEvaluator()
			.setMetricName("r2")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)

		return Array(rmse_lr, r2_lr)
   }

   def applyGeneralizedLinearRegressionModel(normTrainingData:DataFrame , normTestData:DataFrame) : Array[Double] = {
		val glr = new GeneralizedLinearRegression()
  			.setFamily("gaussian")
			.setFeaturesCol("normFeatures")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setRegParam(0.8) //check values for this function
		
		val glrModel = glr.fit(normTrainingData)
		val predictions = glrModel.transform(normTestData)
		val rmse_glr = new RegressionEvaluator()
			.setMetricName("rmse")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)
		
		val r2_glr = new RegressionEvaluator()
			.setMetricName("r2")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)

		return Array(rmse_glr, r2_glr)
   }

	def applyRandomForestRegressionModel(normTrainingData:DataFrame , normTestData:DataFrame) : Array[Double] = {
		val rfr = new RandomForestRegressor()
			.setFeaturesCol("normFeatures")
			.setLabelCol("ArrDelay")
			
		val rfrModel = rfr.fit(normTrainingData)

		val predictions = rfrModel.transform(normTestData)
		val rmse_rfr = new RegressionEvaluator()
			.setMetricName("rmse")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)

		val r2_rfr = new RegressionEvaluator()
			.setMetricName("r2")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)

		return Array(rmse_rfr, r2_rfr)
	}

	def applyGradientBoostedRegressionModel(normTrainingData:DataFrame , normTestData:DataFrame) : Array[Double] = {
		val gbtr = new GBTRegressor()
			.setFeaturesCol("normFeatures")
			.setLabelCol("ArrDelay")
			.setMaxIter(10)
			
		val gbtrModel = gbtr.fit(normTrainingData)
		val predictions = gbtrModel.transform(normTestData)
		val rmse_gbtr = new RegressionEvaluator()
			.setMetricName("rmse")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)

		val r2_gbtr = new RegressionEvaluator()
			.setMetricName("r2")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)
		
		return Array(rmse_gbtr, r2_gbtr)
	}
}
