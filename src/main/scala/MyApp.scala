package upm.bd
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.when
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import java.io.PrintWriter
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.UnivariateFeatureSelector
import org.apache.spark.ml.feature.VarianceThresholdSelector
import org.apache.spark.ml.Pipeline

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
		"Month"
	)
		
	// Numerical Variables
	val IntColumns = Array(
		"Year",
		//"Month",
		//"DayOfMonth",
		"DepTime",
		"CRSDepTime",
		"CRSArrTime",
		"FlightNum",
		"CRSElapsedTime",
		"DepDelay",
		"Distance"
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

		// Drop all excluded columns	
		data = data.drop((ForbiddenColumns++ExcludedColumns): _*)

		

		// TODO: Use the string columns (Origin, destination, UniqueCarrier, DayofMonth) by converting them to categorical variables and find a way to utilize them. 
		val indexer = new StringIndexer()
			.setInputCols(CatColumns)
			.setOutputCols(Array("Ind_Origin","Ind_Dest","Ind_UniqueCarrier","Ind_DayofMonth","Ind_Season"))
		val indexed = indexer.fit(data).transform(data)

		val encoder = new OneHotEncoder()
			.setInputCols(Array("Ind_Origin","Ind_Dest","Ind_UniqueCarrier","Ind_DayofMonth","Ind_Season"))
			.setOutputCols(Array("Enc_Origin","Enc_Dest","Enc_UniqueCarrier","Enc_DayofMonth","Enc_Season"))
		var encoded = encoder.fit(indexed).transform(indexed)
		data = encoded

		// Cast int columns to int
		for (colName <- IntColumns++Array(LabelColumn))
			data = data.withColumn(colName, col(colName).cast("integer"))

		// TODO: remove the 4 lines below before the final submission
		data.printSchema()
		val count = data.count()
		val stringData = data.collect().mkString(" ")
		new PrintWriter("csv_output") { write("NumberOfTotalRows="+count+"\n"+stringData+"\n"); close }

		// --------------------------------------------------------------------------------------------------
		// Machine learning begins
		// --------------------------------------------------------------------------------------------------

		//SPLITING DATA
		val split = data.randomSplit(Array(0.8,0.2))
		val trainingData = split(0)
		val testData = split(1)

		// TODO: Check each variable one by one to determine the relationship with the result of the machine learning algorithm
		applyLinearRegressionModel(trainingData);

		// TODO: Cross validation for the machine learning output

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
}
