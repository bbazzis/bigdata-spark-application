package upm.bd
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql
import org.apache.spark.sql._
import org.apache.spark.SparkContext._
import org.apache.log4j.{Level, Logger}
import java.io.PrintWriter

object MyApp {
	def main(args : Array[String]) {
		Logger.getLogger("org").setLevel(Level.WARN)


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

val EXCLUDED_COLUMNS = Array(
"Cancelled",
"CancellationCode",
"TailNum",
"DayOfWeek",
"TaxiOut"
)

        //	val conf = new SparkConf().setAppName("Spark Assignment 1")
	//	val sc = new SparkContext(conf)



		val spark = org.apache.spark.sql.SparkSession
			.builder()
			.appName("Spark SQL example")
			.config("some option", "value")
			.enableHiveSupport()
			.getOrCreate()

        	val data = spark.read.csv("/tmp/csv/1987.csv")
	        val droppedData = data.drop((FORBIDDEN_COLUMNS++EXCLUDED_COLUMNS): _*)

		val count = droppedData.count()
		val stringData = droppedData.collect().mkString(" ")
		new PrintWriter("csv_output") { write("NumberOfTotalRows="+count+"\n"+stringData+"\n"); close }

	}
}
    
