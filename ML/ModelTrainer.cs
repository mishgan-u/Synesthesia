#nullable enable

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

/// <summary>
/// Trains the two mood-regression models used by MusicColorApp (Valence and Arousal).
/// ML.NET has no multi-output regression out of the box, so two identical FastTree
/// pipelines are trained on the same four audio features and saved separately.
/// </summary>
public class ModelTrainer
{
    private const string FeaturesColumn = "Features";

    private const string ValenceLabel = nameof(TrainData.Valence);
    private const string ArousalLabel = nameof(TrainData.Arousal);

    /// <summary>
    /// Trains both models from <paramref name="csvPath"/> and writes
    /// valence.zip and arousal.zip into <paramref name="outputDirectory"/>.
    /// </summary>
    public void Train(string csvPath, string outputDirectory)
    {
        if (string.IsNullOrWhiteSpace(csvPath))
            throw new ArgumentException("CSV path is required.", nameof(csvPath));

        if (!File.Exists(csvPath))
            throw new FileNotFoundException("Training CSV not found.", csvPath);

        if (string.IsNullOrWhiteSpace(outputDirectory))
            throw new ArgumentException("Output directory is required.", nameof(outputDirectory));

        var mlContext = new MLContext(seed: 0);

        IDataView trainData = mlContext.Data.LoadFromTextFile<TrainData>(
            csvPath,
            separatorChar: ',',
            hasHeader: true);

        // 80/20 split; the same split is used for both targets so metrics are comparable.
        var split = mlContext.Data.TrainTestSplit(trainData, testFraction: 0.2, seed: 0);

        ITransformer valenceModel = BuildPipeline(mlContext, ValenceLabel).Fit(split.TrainSet);
        ITransformer arousalModel = BuildPipeline(mlContext, ArousalLabel).Fit(split.TrainSet);

        Evaluate(mlContext, valenceModel, split.TestSet, ValenceLabel);
        Evaluate(mlContext, arousalModel, split.TestSet, ArousalLabel);

        Directory.CreateDirectory(outputDirectory);

        string valencePath = Path.Combine(outputDirectory, "valence.zip");
        string arousalPath = Path.Combine(outputDirectory, "arousal.zip");

        mlContext.Model.Save(valenceModel, trainData.Schema, valencePath);
        mlContext.Model.Save(arousalModel, trainData.Schema, arousalPath);

        Console.WriteLine($"Saved valence model: {valencePath}");
        Console.WriteLine($"Saved arousal model: {arousalPath}");
    }

    /// <summary>Features -> Features, mean/variance normalisation, FastTree regression on one target.</summary>
    private static IEstimator<ITransformer> BuildPipeline(MLContext mlContext, string labelColumnName)
    {
        return mlContext.Transforms
            .Concatenate(
                FeaturesColumn,
                nameof(TrainData.Bpm),
                nameof(TrainData.SpectralCentroid),
                nameof(TrainData.Rms),
                nameof(TrainData.IsMajor))
            .Append(mlContext.Transforms.NormalizeMeanVariance(FeaturesColumn))
            .Append(mlContext.Regression.Trainers.FastTree(
                labelColumnName: labelColumnName,
                featureColumnName: FeaturesColumn));
    }

    private static void Evaluate(MLContext mlContext, ITransformer model, IDataView testData, string labelColumnName)
    {
        IDataView predictions = model.Transform(testData);
        RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: labelColumnName);

        Console.WriteLine(
            $"{labelColumnName,-8} RMSE: {metrics.RootMeanSquaredError:F4}   R2: {metrics.RSquared:F4}");
    }

    /// <summary>CSV row schema: Bpm,SpectralCentroid,Rms,IsMajor,Valence,Arousal.</summary>
    private sealed class TrainData
    {
        [LoadColumn(0)] public float Bpm { get; set; }

        [LoadColumn(1)] public float SpectralCentroid { get; set; }

        [LoadColumn(2)] public float Rms { get; set; }

        [LoadColumn(3)] public float IsMajor { get; set; }

        [LoadColumn(4)] public float Valence { get; set; }

        [LoadColumn(5)] public float Arousal { get; set; }
    }
}
