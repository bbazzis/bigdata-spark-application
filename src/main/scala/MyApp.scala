package upm.bd
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, when, concat, lit, unix_timestamp, udf}
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import java.io.PrintWriter
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GeneralizedLinearRegression
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
		"DayOfWeek",
		"TaxiOut",
		"Year",
		"Month",
		"DayOfMonth"
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
		"Season"
	)

	val LabelColumn = "ArrDelay"

	def main(args : Array[String]) {
		Logger.getLogger("org").setLevel(Level.WARN)
		
		if (args.length == 0) {
			println("Please enter the directory of the data file")
			System.exit(0)
		}

		val filename = args(0)
		println("filename = " + filename)
		// filename = "/tmp/csv/1987_min.csv"
		// Check whether the file exists at the location

		// Check if the file is of correct type

		// Check if file can be read without an issue
		


		val spark = org.apache.spark.sql.SparkSession
			.builder()
			.appName("Group 13 - Big Data Assignment - Flight Delay Predictor")
			.config("some option", "value") // TODO: Remove or fix
			.enableHiveSupport()
			.getOrCreate()
		
		// Skip malformed data
		// TODO: Probably need to read into RDD(s)
		var data = spark.read.option("header",true).option("mode", "DROPMALFORMED").csv(filename)
		
		// Transform cancelled field to int to evaluate the flight status
		data = data.withColumn("Cancelled", col("Cancelled").cast("integer"))

		/*
		** Remove the rows where "cancelled" field has a value, 
		** so that we don't try to evaluate the flights that did not happen
		*/
		data = data.filter("Cancelled == 0")

		//Transform Month variable into Season
		data = data.withColumn("Month", col("Month").cast("string"))
		//data.show()
		data = data.withColumn("Season", when(col("Month") === 12 || col("Month") === 1 || col("Month") === 2,"Winter")
			.when(col("Month") === 3 || col("Month") === 4 || col("Month") === 5,"Spring")
			.when(col("Month") === 6 || col("Month") === 7 || col("Month") === 8,"Summer")
			.otherwise("Autumn"))

		// Using string columns as categorical data
		val indexer = new StringIndexer()
			.setInputCols(CatColumns)
			.setOutputCols(Array("Ind_Origin","Ind_Dest","Ind_UniqueCarrier","Ind_DayofMonth","Ind_Season"))
		val indexed = indexer.fit(data).transform(data)

		val encoder = new OneHotEncoder()
			.setInputCols(Array("Ind_Origin","Ind_Dest","Ind_UniqueCarrier","Ind_DayofMonth","Ind_Season"))
			.setOutputCols(Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
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

		// TODO: Consider better ways of feature selection
		applyLinearRegressionModel(trainingData);
		applyGeneralizedLinearRegressionModel(trainingData, testData)
	}

	def applyUnivariateFilter( data:DataFrame ) : Unit = {
		val assembler = new VectorAssembler()
			.setInputCols(IntColumns++Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
			.setOutputCol("features")
			.setHandleInvalid("skip")
		
		println("Output of assembler")
		val output = assembler.transform(data)

		val selector = new UnivariateFeatureSelector()
			.setFeatureType("continuous")
			.setLabelType("categorical")
			.setSelectionMode("numTopFeatures")
			.setSelectionThreshold(10)
			.setFeaturesCol("features")
			.setLabelCol(LabelColumn)
			.setOutputCol("selectedFeatures")

		val result = selector.fit(output).transform(output)

		println(s"UnivariateFeatureSelector output with top ${selector.getSelectionThreshold}" +
		s" features selected using f_classif")
		result.select("selectedFeatures").show()
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

   def applyLinearRegressionModel( data:DataFrame ) : Unit = {
	   val assembler = new VectorAssembler()
  			.setInputCols(IntColumns++Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
			.setOutputCol("features")
			.setHandleInvalid("skip")
		
		// println("Output of assembler")
		val output = assembler.transform(data)
		// output.show(truncate=false)
		
		//NORMALIZATION
		val normalizer = new Normalizer()
  			.setInputCol("features")
  			.setOutputCol("normFeatures")
  			.setP(1.0)

		// println("Output of normalizer")
		val l1NormData = normalizer.transform(output)
		// l1NormData.show(truncate=false)

		val lr = new LinearRegression()
  			.setFeaturesCol("normFeatures")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setElasticNetParam(0.8)
		
		//val lrModel = lr.fit(trainingData)
		val lrModel = lr.fit(l1NormData)
		
		println(s"Coefficients: ${lrModel.coefficients}")
		println(s"Intercept: ${lrModel.intercept}")
		val trainingSummary = lrModel.summary
		println(s"numIterations: ${trainingSummary.totalIterations}")
		println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
		trainingSummary.residuals.show()
		println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
		println(s"r2: ${trainingSummary.r2}")
		
   }

   def applyLinearRegressionModelViaPipeline( trainingData:DataFrame, testData:DataFrame ) : Unit = {
	   val assembler = new VectorAssembler()
  			.setInputCols(IntColumns++Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
			.setOutputCol("features")
			.setHandleInvalid("skip")
		
		val normalizer = new Normalizer()
  			.setInputCol("features")
  			.setOutputCol("normFeatures")
  			.setP(1.0)

		val lr = new LinearRegression()
  			.setFeaturesCol("normFeatures")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setElasticNetParam(0.8)
		
		val pipeline = new Pipeline()
			.setStages(Array(assembler, normalizer, lr))
		val model = pipeline.fit(trainingData)
		println("Output of pipeline model for test data")
		model.transform(testData).show(truncate=false)
   }

   def applyGeneralizedLinearRegressionModel( training_data:DataFrame , test_data:DataFrame ) : Unit = {
	   val assembler = new VectorAssembler()
  			.setInputCols(IntColumns++Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
			.setOutputCol("features")
			.setHandleInvalid("skip")
		
		// println("Output of assembler")
		val training_output = assembler.transform(training_data)
		val test_output = assembler.transform(test_data)
		// output.show(truncate=false)
		
		//NORMALIZATION
		val normalizer = new Normalizer()
  			.setInputCol("features")
  			.setOutputCol("normFeatures")
  			.setP(1.0)

		// println("Output of normalizer")
		val gl1NormTrainingData = normalizer.transform(training_output)
		// l1NormData.show(truncate=false)
		val gl1NormTestData = normalizer.transform(test_output)

		val glr = new GeneralizedLinearRegression()
  			.setFamily("gaussian")
			.setFeaturesCol("normFeatures")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setRegParam(0.8) //check values for this function
		
		//val lrModel = lr.fit(trainingData)
		val glrModel = glr.fit(gl1NormTrainingData)
		
		println(s"Coefficients: ${glrModel.coefficients}")
		println(s"Intercept: ${glrModel.intercept}")
		val trainingSummary = glrModel.summary
		println(s"Dispersion: ${trainingSummary.dispersion}")
		println(s"Deviance: ${trainingSummary.deviance}")
		trainingSummary.residuals.show()
		println(s"AIC: ${trainingSummary.aic}")
		println(s"Residual Degree of freedom Null: ${trainingSummary.residualDegreeOfFreedomNull}")
   
		val predictions = glrModel.transform(gl1NormTestData)
		val rmse_glr = new RegressionEvaluator()
			.setMetricName("rmse")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)
		println(s"RMSE: ${rmse_glr}")

		
		val r2_glr = new RegressionEvaluator()
			.setMetricName("r2")
			.setLabelCol("ArrDelay")
			.setPredictionCol("prediction")
			.evaluate(predictions)
		println(s"r2: ${r2_glr}")

   }
}
