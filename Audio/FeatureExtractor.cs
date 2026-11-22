using System;
using System.Collections.Generic;
using NAudio.Wave;


public class FeatureExtractor
{
    private const int FftWindowSize = 4096;
    private const int HopSize = 2048;

    private const int MaxSeconds = 60;

    private const int ReadBlockSize = 16384;

    private const float DefaultBpm = 120f;

    private const double MinTempoCorrelation = 0.3;

    
    private const double MinEnvelopeVariation = 0.15;

    private const double MinBinFrequency = 80.0;
    private const double MaxBinFrequency = 2000.0;

    public AudioFeatures Extract(string filePath)
    {
        float[] mono = ReadMono(filePath, MaxSeconds, out int sampleRate);
        if (mono.Length == 0)
            throw new InvalidOperationException("Аудиофайл пустой");

        float rms = CalculateRms(mono);
        var (spectralCentroid, chroma) = CalculateSpectralAndChroma(mono, sampleRate);
        float bpm = DetectBpm(mono, sampleRate);
        float isMajor = DetectKey(chroma) ? 1f : 0f;

        return new AudioFeatures
        {
            Bpm = bpm,
            SpectralCentroid = spectralCentroid,
            Rms = rms,
            IsMajor = isMajor,
        };
    }

   
    private static float[] ReadMono(string filePath, int maxSeconds, out int sampleRate)
    {
        using var reader = new AudioFileReader(filePath);

        sampleRate = reader.WaveFormat.SampleRate;
        int channels = Math.Max(1, reader.WaveFormat.Channels);
        long sampleLimit = (long)sampleRate * channels * maxSeconds;

        ISampleProvider provider = reader;

        var block = new float[ReadBlockSize * channels];
        var collected = new List<float>(1 << 16);
        long total = 0;

        while (total < sampleLimit)
        {
            int want = (int)Math.Min(block.Length, sampleLimit - total);
            int read = provider.Read(block.AsSpan(0, want));
            if (read <= 0)
                break;

           
            int framesInBlock = read / channels;
            for (int frame = 0; frame < framesInBlock; frame++)
            {
                int start = frame * channels;
                double sum = 0;
                for (int c = 0; c < channels; c++)
                    sum += block[start + c];

                collected.Add((float)(sum / channels));
            }

            total += read;
        }

        return collected.ToArray();
    }

    private static float CalculateRms(float[] samples)
    {
        if (samples.Length == 0)
            return 0f;

        double sumOfSquares = 0;
        foreach (float sample in samples)
            sumOfSquares += (double)sample * sample;

        return (float)Math.Sqrt(sumOfSquares / samples.Length);
    }

    
    private static (float centroid, double[] chroma) CalculateSpectralAndChroma(float[] samples, int sampleRate)
    {
        var chroma = new double[12];
        if (samples.Length == 0 || sampleRate <= 0)
            return (0f, chroma);

        var hann = new FftSharp.Windows.Hanning();
        var frame = new double[FftWindowSize];
        int halfBins = FftWindowSize / 2;

        double weightedSum = 0;
        double totalMagnitude = 0;

        for (int offset = 0; offset < samples.Length; offset += HopSize)
        {
            int available = Math.Min(FftWindowSize, samples.Length - offset);
            Array.Clear(frame, 0, frame.Length);
            for (int i = 0; i < available; i++)
                frame[i] = samples[offset + i];

            double[] windowed = hann.Apply(frame);
            double[] magnitude = FftSharp.FFT.Magnitude(FftSharp.FFT.Forward(windowed));
            int bins = Math.Min(halfBins, magnitude.Length);

            for (int bin = 1; bin < bins; bin++)
            {
                double frequency = bin * (double)sampleRate / FftWindowSize;
                double mag = magnitude[bin];

                weightedSum += frequency * mag;
                totalMagnitude += mag;

                if (frequency < MinBinFrequency || frequency > MaxBinFrequency)
                    continue;

                int pitchClass = (int)Math.Round(69 + 12 * Math.Log2(frequency / 440.0)) % 12;
                if (pitchClass < 0)
                    pitchClass += 12;

                chroma[pitchClass] += mag;
            }

           
            if (available < FftWindowSize)
                break;
        }

        float centroid = totalMagnitude > 0 ? (float)(weightedSum / totalMagnitude) : 0f;

        double chromaSum = 0;
        for (int i = 0; i < chroma.Length; i++)
            chromaSum += chroma[i];

        if (chromaSum > 0)
            for (int i = 0; i < chroma.Length; i++)
                chroma[i] /= chromaSum;

        return (centroid, chroma);
    }

    
    private static float DetectBpm(float[] samples, int sampleRate)
    {
        const int envelopeFrame = 1024;
        const int envelopeHop = 512;

        if (sampleRate <= 0 || samples.Length < envelopeFrame * 2)
            return DefaultBpm;

        int frameCount = (samples.Length - envelopeFrame) / envelopeHop + 1;
        var envelope = new double[frameCount];

        for (int frame = 0; frame < frameCount; frame++)
        {
            int start = frame * envelopeHop;
            double sum = 0;
            for (int i = 0; i < envelopeFrame; i++)
            {
                double sample = samples[start + i];
                sum += sample * sample;
            }

            envelope[frame] = Math.Sqrt(sum / envelopeFrame);
        }

        double mean = 0;
        for (int i = 0; i < frameCount; i++)
            mean += envelope[i];
        mean /= frameCount;

        double variance = 0;
        for (int i = 0; i < frameCount; i++)
        {
            double deviation = envelope[i] - mean;
            variance += deviation * deviation;
        }
        variance /= frameCount;

       
        if (mean <= 0 || Math.Sqrt(variance) < MinEnvelopeVariation * mean)
            return DefaultBpm;

        for (int i = 0; i < frameCount; i++)
            envelope[i] -= mean;

        double envelopeRate = (double)sampleRate / envelopeHop;
        int minLag = Math.Max(1, (int)Math.Floor(envelopeRate * 60.0 / 200.0));
        int maxLag = Math.Min(frameCount - 1, (int)Math.Ceiling(envelopeRate * 60.0 / 60.0));

        if (maxLag <= minLag)
            return DefaultBpm;

        double bestScore = 0;
        int bestLag = 0;

        for (int lag = minLag; lag <= maxLag; lag++)
        {
            double cross = 0;
            double energyA = 0;
            double energyB = 0;

            for (int i = 0; i + lag < frameCount; i++)
            {
                cross += envelope[i] * envelope[i + lag];
                energyA += envelope[i] * envelope[i];
                energyB += envelope[i + lag] * envelope[i + lag];
            }

            double norm = Math.Sqrt(energyA * energyB);
            double score = norm > 0 ? cross / norm : 0;

            if (score > bestScore)
            {
                bestScore = score;
                bestLag = lag;
            }
        }

        if (bestLag == 0 || bestScore < MinTempoCorrelation)
            return DefaultBpm;

        double bpm = 60.0 * envelopeRate / bestLag;
        while (bpm < 60.0)
            bpm *= 2.0;
        while (bpm > 200.0)
            bpm /= 2.0;

        return (float)Math.Round(bpm, 1);
    }

    /// <summary>Krumhansl-Schmuckler style profile correlation; true means major.</summary>
    private static bool DetectKey(double[] chroma)
    {
        double[] majorProfile = { 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88 };
        double[] minorProfile = { 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17 };

        double majorCorr = Correlation(chroma, majorProfile);
        double minorCorr = Correlation(chroma, minorProfile);

        return majorCorr >= minorCorr;
    }

    private static double Correlation(double[] a, double[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        if (n == 0)
            return 0;

        double sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
        for (int i = 0; i < n; i++)
        {
            sumA += a[i];
            sumB += b[i];
            sumAB += a[i] * b[i];
            sumA2 += a[i] * a[i];
            sumB2 += b[i] * b[i];
        }

        double numerator = n * sumAB - sumA * sumB;
        double denominator = Math.Sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));

        return denominator == 0 ? 0 : numerator / denominator;
    }
}