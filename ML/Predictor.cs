#nullable enable

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

/// <summary>
/// Valence/arousal regression output of the two trained models.
/// </summary>
public class EmotionPrediction
{
    public float Valence { get; set; }

    public float Arousal { get; set; }
}

/// <summary>
/// Loads the two trained regression models (models/valence.zip, models/arousal.zip)
/// and maps <see cref="AudioFeatures"/> to an <see cref="EmotionPrediction"/>.
/// </summary>
public sealed class Predictor : IDisposable
{
    private const string ModelsFolderName = "models";
    private const string ValenceModelFile = "valence.zip";
    private const string ArousalModelFile = "arousal.zip";
    private const string ScoreColumnName = "Score";

    /// <summary>
    /// Default model location: the "models" folder next to the executable, never the
    /// current working directory (Avalonia can be started from anywhere).
    /// </summary>
    private static readonly string DefaultModelsDirectory =
        Path.Combine(AppContext.BaseDirectory, ModelsFolderName);

    // PredictionEngine is not thread-safe; both engines share one lock.
    private readonly object _sync = new object();

    private readonly PredictionEngine<FeaturesInput, ScoreOutput> _valenceEngine;
    private readonly PredictionEngine<FeaturesInput, ScoreOutput> _arousalEngine;

    /// <param name="modelsDirectory">
    /// Optional explicit folder holding valence.zip and arousal.zip (absolute or
    /// relative to the current directory). When null, the "models" folder next to
    /// the executable is used.
    /// </param>
    public Predictor(string? modelsDirectory = null)
    {
        string directory = string.IsNullOrWhiteSpace(modelsDirectory)
            ? DefaultModelsDirectory
            : Path.GetFullPath(modelsDirectory);

        string valencePath = Path.Combine(directory, ValenceModelFile);
        string arousalPath = Path.Combine(directory, ArousalModelFile);

        EnsureModelExists(valencePath);
        EnsureModelExists(arousalPath);

        var mlContext = new MLContext(seed: 0);

        ITransformer valenceModel = mlContext.Model.Load(valencePath, out _);
        ITransformer arousalModel = mlContext.Model.Load(arousalPath, out _);

        _valenceEngine = mlContext.Model.CreatePredictionEngine<FeaturesInput, ScoreOutput>(valenceModel);
        _arousalEngine = mlContext.Model.CreatePredictionEngine<FeaturesInput, ScoreOutput>(arousalModel);
    }

    /// <summary>Predicts the mood coordinates, clamped to [0, 1].</summary>
    public EmotionPrediction Predict(AudioFeatures features)
    {
        ArgumentNullException.ThrowIfNull(features);

        var input = new FeaturesInput
        {
            Bpm = features.Bpm,
            SpectralCentroid = features.SpectralCentroid,
            Rms = features.Rms,
            IsMajor = features.IsMajor,
        };

        float valence;
        float arousal;

        lock (_sync)
        {
            valence = _valenceEngine.Predict(input).Score;
            arousal = _arousalEngine.Predict(input).Score;
        }

        return new EmotionPrediction
        {
            Valence = Math.Clamp(valence, 0f, 1f),
            Arousal = Math.Clamp(arousal, 0f, 1f),
        };
    }

    public void Dispose()
    {
        _valenceEngine.Dispose();
        _arousalEngine.Dispose();
    }

    private static void EnsureModelExists(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Model file not found: '{path}'. " +
                "Train the models or copy valence.zip/arousal.zip into the 'models' folder next to the executable.",
                path);
    }

    /// <summary>Model input: the four features the pipelines were trained on.</summary>
    private sealed class FeaturesInput
    {
        public float Bpm { get; set; }

        public float SpectralCentroid { get; set; }

        public float Rms { get; set; }

        public float IsMajor { get; set; }
    }

    /// <summary>Model output: FastTree writes its prediction into "Score".</summary>
    private sealed class ScoreOutput
    {
        [ColumnName(ScoreColumnName)]
        public float Score { get; set; }
    }
}