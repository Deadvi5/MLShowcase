using Microsoft.ML;

const string trainingFilePath = "../../../test.tsv";
const string modelFilePath = "../../../model.zip";
const string inputColumnName = "Result";
const string outputColumnName = "Label";

const string featurizedInputColumnName = "Input";
const string featurizedOutputColumnName = "Featurized";
const string contcatenatedInputColumnName = "Features";


var mlContext = new MLContext(seed: 0);
PredictionEngine<InputModel, OutputModel> predictionEngine;
ITransformer model;

Console.WriteLine("Do you want to retrain the model? y/n (default n)");
var retrainModel = Console.ReadLine() ?? "n";

if (retrainModel == "y")
{
    var trainingDataView = mlContext.Data.LoadFromTextFile<InputModel>(trainingFilePath, hasHeader: true);

    var pipeline = BuildTrainingPipeline();
    TrainModel(trainingDataView, pipeline);
    SaveModelAsFile(trainingDataView);
}

while (true)
{
    Console.WriteLine("Write anime character name to find the show");
    var subject = Console.ReadLine();
    var result = PredictDepartmentForSubject(subject ?? string.Empty);
    Console.WriteLine("Result is: " + result);
}

string PredictDepartmentForSubject(string userInput)
{
    var transformer = mlContext.Model.Load(modelFilePath, out _);
    var inputModel = new InputModel { Input = userInput };
    predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(transformer);
    var result = predictionEngine.Predict(inputModel);
    return result?.Output ?? string.Empty;
}

void SaveModelAsFile(IDataView trainingDataView)
{
    mlContext.Model.Save(model, trainingDataView.Schema, modelFilePath);
}

IEstimator<ITransformer> BuildTrainingPipeline()
{
    var processPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: inputColumnName, outputColumnName: outputColumnName)
        .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: featurizedInputColumnName, outputColumnName:featurizedOutputColumnName))
        .Append(mlContext.Transforms.Concatenate(contcatenatedInputColumnName, featurizedOutputColumnName))
        .AppendCacheCheckpoint(mlContext);

    return processPipeline;
}

void TrainModel(IDataView trainingDataView, IEstimator<ITransformer> buildPipeline)
{
    var trainingPipeline = buildPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    model = trainingPipeline.Fit(trainingDataView);
}