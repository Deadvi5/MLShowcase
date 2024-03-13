using Microsoft.ML.Data;

public class InputModel
{
    [LoadColumn(0)]
    public string? Input { get; set; }
    [LoadColumn(1)]
    public string? Result { get; set; }
}