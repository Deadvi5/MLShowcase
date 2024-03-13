using Microsoft.ML.Data;

public class OutputModel
{
    [ColumnName("PredictedLabel")]
    public string? Output { get; set; }
}