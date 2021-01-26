import org.apache.spark._ 
import org.apache.spark.SparkContext._ 
import org.apache.log4j._ 
import org.apache.spark.sql.functions._ 
import org.apache.spark.sql.SparkSession 
import org.apache.spark.ml.regression.LinearRegression 
import org.apache.hadoop.fs.DF 
import org.apache.spark.ml.feature.VectorAssembler 
import org.apache.spark.ml.evaluation.RegressionEvaluator 

 
object TaxiAnalysis extends Serializable {   

  val MILES_MIN = 0  // minimum miles 

  val MILES_MAX = 100  // maximum miles 

  val DURATION_MAX = 1500  // maximum seconds 

  val FARE_MAX = 200  // maximum seconds 

   
  def main(args: Array[String]) = { 
    Logger.getLogger("org").setLevel(Level.ERROR) 

    val spark = SparkSession 
      .builder 
      .appName("TaxiLinearRegression") 
      .master("local") 
      .getOrCreate() 

    // import spark.implicits._ 
    var df = spark.read 
      .option("header", "true") 
      .option("inferSchema", "true") 
      .csv("/Users/mayowa/Downloads/Taxi_Trips.csv")   

    println("\n---------------------------------------------------------") 

    println("SCHEMA") 

    println("---------------------------------------------------------") 

    df.printSchema() 

    // Data cleaning 
    // Remove rows having null value for any of the column 
    df = df.na.drop() 

    // Remove less frequent values 
    df = df.filter(df("Trip_Miles") > MILES_MIN  &&  df("Trip_Miles") <= MILES_MAX  &&  df("Trip_Seconds") < DURATION_MAX  &&  df("Fare") < FARE_MAX).toDF() 

     

    // Rename column "Fare" to "Label". This is required for Spark ML Library 
    df = df.withColumnRenamed("Fare", "label") 

     

    // Use Trip Seconds and Trip miles as independent variables 
    var assembler = new VectorAssembler() 
      .setInputCols(Array("Trip_Seconds", "Trip_Miles")) 
      .setOutputCol("features") 

       

    df = assembler.transform(df) 
    // Create a Linear Regression model and fit to training data set 

    var lr = new LinearRegression() 
      .setMaxIter(10) 
      .setRegParam(0.0) 

      //.setElasticNetParam(0.8) 

    // Split data into 70% training and 30% test data 
    var Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed=5) 

    // Fit linear model 
    var lrModel_all = lr.fit(train)     
    var lrPred_all = lrModel_all.transform(test) 

    val evaluator = new RegressionEvaluator() 
      .setMetricName("rmse") 
      .setLabelCol("label") 
      .setPredictionCol("prediction") 

   
    println("\nLINEAR MODEL: PREDICTING FARE AS A FUNCTION OF TRIP MILES AND TRIP DURATION") 

    println("---------------------------------------------------------") 

    
    var trainingSummary = lrModel_all.summary 
    println(s"r2: ${trainingSummary.r2}") 
    println(s"Coefficients: ${lrModel_all.coefficients}") 
    println(s"Intercept: ${lrModel_all.intercept}") 
  

    var rmse = evaluator.evaluate(lrPred_all) 
    println(s"Root-mean-square error = $rmse" + "\n") 
    
    println("\nPREDICTIONS") 

    println("---------------------------------------------------------") 
    lrPred_all.select("prediction", "label", "features").show(10) 
    spark.stop() 

  } 

} 