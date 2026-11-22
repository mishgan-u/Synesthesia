#nullable enable

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

/// <summary>
/// Builds a CSV training set (features + mood labels) for the MusicColorApp model.
/// Two modes: fully synthetic features, or features extracted from a folder of
/// real audio files via <see cref="FeatureExtractor"/>.
/// </summary>
public class SyntheticDataGenerator
{
    /// <summary>Exact CSV header expected by the trainer.</summary>
    public const string CsvHeader = "Bpm,SpectralCentroid,Rms,IsMajor,Valence,Arousal";

    private const int ProgressEvery = 500;
    private const int DefaultSeed = 42;

    private const double NoiseSigma = 0.05;

    private const double BpmLow = 60.0;
    private const double BpmHigh = 180.0;
    private const double CentroidLow = 500.0;
    private const double CentroidHigh = 4000.0;
    private const double RmsLow = 0.05;
    private const double RmsHigh = 1.0;

    private readonly FeatureExtractor _extractor = new FeatureExtractor();

    /// <summary>Mode A: uniform random features, labels derived by the mood rules.</summary>
    public void GenerateFromSynthetic(string outputPath, int rowCount = 3000, int seed = 42)
    {
        if (rowCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(rowCount), rowCount, "Row count must be positive.");

        var random = new Random(seed);

        using var writer = CreateWriter(outputPath);
        writer.WriteLine(CsvHeader);

        for (int row = 1; row <= rowCount; row++)
        {
            AudioFeatures features = NextSyntheticFeatures(random);
            WriteRow(writer, features, random);

            if (row % ProgressEvery == 0)
                Console.WriteLine($"Synthetic: {row}/{rowCount} rows");
        }

        Console.WriteLine($"Synthetic: done, {rowCount} rows -> {Path.GetFullPath(outputPath)}");
    }

    /// <summary>Mode B: features extracted from every *.mp3 / *.wav under <paramref name="folder"/>.</summary>
    public void GenerateFromAudioFolder(string outputPath, string folder, int seed = 42)
    {
        if (string.IsNullOrWhiteSpace(folder))
            throw new ArgumentException("Audio folder is required.", nameof(folder));

        if (!Directory.Exists(folder))
            throw new DirectoryNotFoundException($"Audio folder not found: {folder}");

        var random = new Random(seed);

        List<string> files = Directory
            .EnumerateFiles(Path.GetFullPath(folder), "*", SearchOption.AllDirectories)
            .Where(IsSupportedAudioFile)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToList();

        if (files.Count == 0)
            Console.WriteLine($"Audio: no *.mp3 or *.wav files under {Path.GetFullPath(folder)}");

        int written = 0;
        int skipped = 0;

        using var writer = CreateWriter(outputPath);
        writer.WriteLine(CsvHeader);

        foreach (string file in files)
        {
            AudioFeatures features;
            try
            {
                features = _extractor.Extract(file);
            }
            catch (Exception ex)
            {
                skipped++;
                Console.WriteLine($"Audio: skipped {Path.GetFileName(file)} ({ex.GetType().Name}: {ex.Message})");
                continue;
            }

            WriteRow(writer, features, random);
            written++;
            Console.WriteLine($"Audio: [{written}/{files.Count}] {Path.GetFileName(file)}");
        }

        Console.WriteLine($"Audio: done, {written} rows, {skipped} skipped -> {Path.GetFullPath(outputPath)}");
    }

    /// <summary>
    /// Entry point: Mode B when <paramref name="audioFolder"/> is set, Mode A otherwise.
    /// </summary>
    public void GenerateCsv(string outputPath, int rowCount = 3000, string? audioFolder = null)
    {
        if (!string.IsNullOrWhiteSpace(audioFolder))
            GenerateFromAudioFolder(outputPath, audioFolder!, DefaultSeed);
        else
            GenerateFromSynthetic(outputPath, rowCount, DefaultSeed);
    }

    private static AudioFeatures NextSyntheticFeatures(Random random)
    {
        return new AudioFeatures
        {
            Bpm = (float)Uniform(random, BpmLow, BpmHigh),
            SpectralCentroid = (float)Uniform(random, CentroidLow, CentroidHigh),
            Rms = (float)Uniform(random, RmsLow, RmsHigh),
            IsMajor = random.Next(2) == 0 ? 0f : 1f,
        };
    }

    private static void WriteRow(TextWriter writer, AudioFeatures features, Random random)
    {
        var (valence, arousal) = ComputeLabels(features, random);

        string line = string.Format(
            CultureInfo.InvariantCulture,
            "{0:F4},{1:F4},{2:F4},{3:F4},{4:F4},{5:F4}",
            features.Bpm,
            features.SpectralCentroid,
            features.Rms,
            features.IsMajor,
            valence,
            arousal);

        writer.WriteLine(line);
    }

    private static (double Valence, double Arousal) ComputeLabels(AudioFeatures features, Random random)
    {
        double bpm = features.Bpm;
        double rms = Math.Clamp((double)features.Rms, 0.0, 1.0);
        double centroid = features.SpectralCentroid;
        double isMajor = features.IsMajor;

        double arousal = 0.55 * Normalize(bpm, BpmLow, BpmHigh)
                       + 0.25 * rms
                       + 0.20 * Normalize(centroid, CentroidLow, CentroidHigh);

        double valence = 0.35 * isMajor
                       + 0.25 * rms
                       + 0.20 * Normalize(centroid, CentroidLow, CentroidHigh)
                       + 0.20 * Math.Clamp(1.0 - Math.Abs(bpm - 120.0) / 60.0, 0.0, 1.0);

        arousal = Math.Clamp(arousal + NoiseSigma * NextGaussian(random), 0.0, 1.0);
        valence = Math.Clamp(valence + NoiseSigma * NextGaussian(random), 0.0, 1.0);

        return (valence, arousal);
    }

    private static double Normalize(double value, double low, double high)
        => Math.Clamp((value - low) / (high - low), 0.0, 1.0);

    private static double NextGaussian(Random random)
    {
        // Box-Muller; 1 - NextDouble() keeps the logarithm away from zero.
        double u1 = 1.0 - random.NextDouble();
        double u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    private static double Uniform(Random random, double low, double high)
        => low + random.NextDouble() * (high - low);

    private static bool IsSupportedAudioFile(string path)
    {
        string extension = Path.GetExtension(path);
        return extension.Equals(".mp3", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".wav", StringComparison.OrdinalIgnoreCase);
    }

    private static StreamWriter CreateWriter(string outputPath)
    {
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("Output path is required.", nameof(outputPath));

        string fullPath = Path.GetFullPath(outputPath);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        return new StreamWriter(fullPath, append: false, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }
}
