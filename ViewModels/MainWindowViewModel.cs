#nullable enable

using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using Avalonia.Media;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using NAudio.Wave;

// NAudio's WaveOut output device is Windows-only; Synesthesia is a Windows desktop app.
// (NAudio 3.x renamed the old WaveOutEvent class to WaveOut.)
#pragma warning disable CA1416

namespace Synesthesia.ViewModels;

/// <summary>
/// Drives the main window: asks the view for a file, extracts audio features,
/// predicts valence/arousal, turns them into a display colour and plays the track.
/// </summary>
public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
    private static readonly TimeSpan PositionUpdateInterval = TimeSpan.FromMilliseconds(500);

    private readonly FeatureExtractor _featureExtractor = new FeatureExtractor();

    private Predictor? _predictor;

    private WaveOut? _waveOut;
    private AudioFileReader? _audioReader;
    private string? _currentFilePath;

    private DispatcherTimer? _positionTimer;
    private bool _suppressSeek;

    [ObservableProperty]
    private string _statusMessage = "Ready.";

    [ObservableProperty]
    private IBrush _predictedColorBrush = Brushes.Gray;

    [ObservableProperty]
    private string _bpmText = "BPM: -";

    [ObservableProperty]
    private string _valenceText = "Valence: -";

    [ObservableProperty]
    private string _arousalText = "Arousal: -";

    // Numeric values for the dashboard (big numbers + 0..1 progress indicators).
    [ObservableProperty]
    private double _bpmValue = 0;

    [ObservableProperty]
    private double _valenceValue = 0;

    [ObservableProperty]
    private double _arousalValue = 0;

    [ObservableProperty]
    private string _playButtonIcon = "▶";

    [ObservableProperty]
    private bool _hasTrack = false;

    /// <summary>Slider position in seconds (two-way bound to the slider).</summary>
    [ObservableProperty]
    private double _trackPosition = 0;

    [ObservableProperty]
    private double _trackDuration = 0;

    [ObservableProperty]
    private string _timeDisplay = "00:00 / 00:00";

    /// <summary>
    /// True while the user drags the slider: the position timer stops writing
    /// <see cref="TrackPosition"/>/<see cref="TimeDisplay"/> so the thumb is not pulled back.
    /// </summary>
    [ObservableProperty]
    private bool _isSeeking = false;

    /// <summary>
    /// Set by the view. Shows the OS file picker and returns the selected path,
    /// or null when the user cancels (a view model must not open windows itself).
    /// </summary>
    public Func<Task<string?>>? FilePickerInteraction { get; set; }

    [RelayCommand]
    private async Task LoadTrackAsync()
    {
        if (FilePickerInteraction is null)
        {
            StatusMessage = "File picker is not available.";
            return;
        }

        string? filePath;
        try
        {
            filePath = await FilePickerInteraction();
        }
        catch (Exception ex)
        {
            StatusMessage = $"File selection failed: {ex.Message}";
            return;
        }

        if (string.IsNullOrWhiteSpace(filePath))
        {
            // User cancelled the dialog: keep the previous result on screen.
            return;
        }

        // StatusMessage only ever holds the track name.
        StatusMessage = Path.GetFileName(filePath);

        try
        {
            // Feature extraction is CPU heavy; keep the UI thread responsive.
            AudioFeatures features = await Task.Run(() => _featureExtractor.Extract(filePath));

            EmotionPrediction prediction = GetPredictor().Predict(features);
            Color color = ColorMapper.MapToColor(prediction.Valence, prediction.Arousal);

            // A new track replaces the previous one: stop whatever was playing.
            StopPlayback();

            _currentFilePath = filePath;
            HasTrack = true;

            PredictedColorBrush = new SolidColorBrush(color);
            BpmText = $"BPM: {features.Bpm:F0}";
            ValenceText = $"Valence: {prediction.Valence:F2}";
            ArousalText = $"Arousal: {prediction.Arousal:F2}";

            BpmValue = features.Bpm;
            ValenceValue = prediction.Valence;
            ArousalValue = prediction.Arousal;
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
    }

    [RelayCommand]
    private void PlayPause()
    {
        if (string.IsNullOrEmpty(_currentFilePath) || !File.Exists(_currentFilePath))
            return;

        if (_waveOut is not null && _waveOut.PlaybackState == PlaybackState.Playing)
        {
            StopPlayback();
            return;
        }

        try
        {
            // Replace any previous (paused or already finished) playback.
            DisposePlayback();

            _audioReader = new AudioFileReader(_currentFilePath);
            _waveOut = new WaveOut();
            _waveOut.Init(_audioReader);
            _waveOut.Play();

            PlayButtonIcon = "⏹";
            StartPositionTimer();
        }
        catch (Exception ex)
        {
            // StatusMessage is reserved for the track name, so report to the debugger only.
            Debug.WriteLine($"Playback failed: {ex.Message}");

            DisposePlayback();
            PlayButtonIcon = "▶";
            ResetPosition();
        }
    }

    /// <summary>
    /// Called when the user releases the slider: ends seeking and jumps playback
    /// to <paramref name="newPosition"/> seconds.
    /// </summary>
    [RelayCommand]
    private void Seek(double newPosition)
    {
        IsSeeking = false;
        ApplySeek(newPosition);
    }

    /// <summary>
    /// Slider two-way binding: seek for user/programmatic changes, but ignore the
    /// position written by the timer (guarded by <c>_suppressSeek</c>).
    /// </summary>
    partial void OnTrackPositionChanged(double value)
    {
        if (_suppressSeek)
            return;

        ApplySeek(value);
    }

    private void ApplySeek(double seconds)
    {
        AudioFileReader? reader = _audioReader;
        if (reader is null)
            return;

        try
        {
            double upperBound = Math.Max(TrackDuration, 0);
            reader.CurrentTime = TimeSpan.FromSeconds(Math.Clamp(seconds, 0, upperBound));
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Seek failed: {ex.Message}");
        }
    }

    private void StartPositionTimer()
    {
        StopPositionTimer();

        _positionTimer = new DispatcherTimer { Interval = PositionUpdateInterval };
        _positionTimer.Tick += OnPositionTimerTick;
        _positionTimer.Start();

        UpdatePlaybackPosition();
    }

    private void OnPositionTimerTick(object? sender, EventArgs e) => UpdatePlaybackPosition();

    private void UpdatePlaybackPosition()
    {
        AudioFileReader? reader = _audioReader;
        if (reader is null || _waveOut is null)
        {
            StopPlayback();
            return;
        }

        // Playback finished on its own.
        if (_waveOut.PlaybackState != PlaybackState.Playing)
        {
            StopPlayback();
            return;
        }

        TrackDuration = reader.TotalTime.TotalSeconds;

        // The user owns the thumb while dragging: do not fight it with timer updates.
        if (IsSeeking)
            return;

        _suppressSeek = true;
        TrackPosition = Math.Clamp(reader.CurrentTime.TotalSeconds, 0, Math.Max(TrackDuration, 0));
        _suppressSeek = false;

        TimeDisplay = $"{FormatTime(reader.CurrentTime)} / {FormatTime(reader.TotalTime)}";
    }

    private void StopPlayback()
    {
        StopPositionTimer();

        if (_waveOut is not null)
        {
            try
            {
                _waveOut.Stop();
            }
            catch (Exception)
            {
                // The output device may already be gone; disposing below still applies.
            }
        }

        DisposePlayback();
        PlayButtonIcon = "▶";
        IsSeeking = false;
        ResetPosition();
    }

    private void ResetPosition()
    {
        _suppressSeek = true;
        TrackPosition = 0;
        _suppressSeek = false;
        TimeDisplay = "00:00 / 00:00";
    }

    private void StopPositionTimer()
    {
        DispatcherTimer? timer = _positionTimer;
        _positionTimer = null;

        if (timer is null)
            return;

        timer.Stop();
        timer.Tick -= OnPositionTimerTick;
    }

    private void DisposePlayback()
    {
        WaveOut? waveOut = _waveOut;
        AudioFileReader? audioReader = _audioReader;

        _waveOut = null;
        _audioReader = null;

        waveOut?.Dispose();
        audioReader?.Dispose();
    }

    public void Dispose()
    {
        StopPositionTimer();
        DisposePlayback();
    }

    private static string FormatTime(TimeSpan time)
    {
        if (time < TimeSpan.Zero)
            time = TimeSpan.Zero;

        return time.ToString(@"mm\:ss", CultureInfo.InvariantCulture);
    }

    private Predictor GetPredictor()
        => _predictor ??= new Predictor();
}