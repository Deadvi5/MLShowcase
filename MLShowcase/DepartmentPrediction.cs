using Microsoft.ML.Data;

public class DepartmentPrediction
{
    [ColumnName("PredictedLabel")]
    public string? Department { get; set; }
}