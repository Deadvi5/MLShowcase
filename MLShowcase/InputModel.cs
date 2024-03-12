using Microsoft.ML.Data;

public class InputModel
{
    [LoadColumn(0)]
    public string Character { get; set; }
    [LoadColumn(1)]
    public string Alignment { get; set; }
}