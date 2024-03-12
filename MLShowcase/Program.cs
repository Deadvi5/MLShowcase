using Microsoft.ML;

var _trainingFilePath = "/Users/villal/Documents/dev/MLShowcase/MLShowcase/MLShowcase/test.tsv";
var _modelFilePath = "/Users/villal/Documents/dev/MLShowcase/MLShowcase/MLShowcase/model.zip";
const string inputColumnName = "Alignment";
const string outputColumnName = "Label";

const string featurizedInputColumnName = "Character";
const string featurizedOutputColumnName = "Featurized";
const string contcatenatedInputColumnName = "Features";


var mlContext = new MLContext(seed: 0);
PredictionEngine<InputModel, DepartmentPrediction> predictionEngine;
ITransformer _model;

Console.WriteLine("Do you want to retrain the model? y/n (default n)");
string retrainModel = Console.ReadLine() ?? "n";

if (retrainModel == "y")
{
    IDataView _trainingDataView = mlContext.Data.LoadFromTextFile<InputModel>(_trainingFilePath, hasHeader: true);

    var pipeline = ProcessData();
    BuildAndTrainModel(_trainingDataView, pipeline);
    SaveModelAsFile(_trainingDataView);
}

while (true)
{
    Console.WriteLine("Write anime character name to find the show");
    var subject = Console.ReadLine();
    var result = PredictDepartmentForSubject(subject ?? string.Empty);
    Console.WriteLine("Result is: " + result);
}

string PredictDepartmentForSubject(string subjectLine)
{
    var model = mlContext.Model.Load(_modelFilePath, out var modelInputSchema);
    var emailSubject = new InputModel { Character = subjectLine };
    predictionEngine = mlContext.Model.CreatePredictionEngine<InputModel, DepartmentPrediction>(model);
    var result = predictionEngine.Predict(emailSubject);
    return result?.Department ?? string.Empty;
}

void SaveModelAsFile(IDataView trainingDataView)
{
    mlContext.Model.Save(_model, trainingDataView.Schema, _modelFilePath);
}

IEstimator<ITransformer> ProcessData()
{
    var processPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: inputColumnName, outputColumnName: outputColumnName)
        .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: featurizedInputColumnName, outputColumnName:featurizedOutputColumnName))
        .Append(mlContext.Transforms.Concatenate(contcatenatedInputColumnName, featurizedOutputColumnName))
        .AppendCacheCheckpoint(mlContext);

    return processPipeline;
}

void BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> buildPipeline)
{
    var trainingPipeline = buildPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    _model = trainingPipeline.Fit(trainingDataView);
}