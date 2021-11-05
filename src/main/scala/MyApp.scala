package upm.bd
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.sql.functions.col
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import java.io.PrintWriter
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.regression.LinearRegression

object MyApp {
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
		
		
		val FORBIDDEN_COLUMNS = Array(
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
// The following columns are exluded because
// they were either mainly null or did not have 
// an effect on the resulting delay

val EXCLUDED_COLUMNS = Array(
"Cancelled",
"CancellationCode",
"TailNum",
"DayOfWeek",
"TaxiOut",
"Origin",
"Dest",
"UniqueCarrier"
)

val INT_COLUMNS = Array(
"Year",
"Month",
"DayOfMonth",
"DepTime",
"CRSDepTime",
"CRSArrTime",
"FlightNum",
"CRSElapsedTime",
"ArrDelay",
"DepDelay",
"Distance"
)



		val spark = org.apache.spark.sql.SparkSession
			.builder()
			.appName("Spark SQL example")
			.config("some option", "value")
			.enableHiveSupport()
			.getOrCreate()
		
		var data = spark.read.option("header",true)
					.csv(filename)
		data = data.withColumn("Cancelled", col("Cancelled").cast("integer"))

		// Remove the rows where "cancelled" field has a value, 
		// so that we don't try to evalueate flights that did not happen
		data = data.filter("Cancelled == 0")		
		data = data.drop((FORBIDDEN_COLUMNS++EXCLUDED_COLUMNS): _*)

		// Cast int columns to int
		for (colName <- INT_COLUMNS)
			data = data.withColumn(colName, col(colName).cast("integer"))

		data.printSchema()

		val count = data.count()
		val stringData = data.collect().mkString(" ")
		new PrintWriter("csv_output") { write("NumberOfTotalRows="+count+"\n"+stringData+"\n"); close }

		val columns = data.columns.toSeq
		val assembler = new VectorAssembler()
  			.setInputCols(Array(columns: _*))
  			.setOutputCol("features")
		
		val output = assembler.transform(data)
			//output.show(truncate=false)

		//NORMALIZATION
		val normalizer = new Normalizer()
  			.setInputCol("features")
  			.setOutputCol("normFeatures")
  			.setP(1.0)

		val l1NormData = normalizer.transform(output)
		//l1NormData.show(truncate=false)

		val lr = new LinearRegression()
  			.setFeaturesCol("features")
  			.setLabelCol("ArrDelay")
  			.setMaxIter(10)
  			.setElasticNetParam(0.8)
		
		val lrModel = lr.fit(output)
		println(s"Coefficients: ${lrModel.coefficients}")
		println(s"Intercept: ${lrModel.intercept}")
		val trainingSummary = lrModel.summary
		println(s"numIterations: ${trainingSummary.totalIterations}")
		println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
		trainingSummary.residuals.show()
		println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
		println(s"r2: ${trainingSummary.r2}")
	}
}
    
