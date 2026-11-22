using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Platform.Storage;
using Synesthesia.ViewModels;

namespace Synesthesia.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        // App.axaml.cs assigns DataContext after this constructor returns, so the
        // file picker is wired up when the data context actually changes.
        DataContextChanged += (_, _) => AttachFilePicker();
    }

    /// <summary>Gives the view model a way to ask the view for a file.</summary>
    private void AttachFilePicker()
    {
        if (DataContext is MainWindowViewModel viewModel)
            viewModel.FilePickerInteraction = PickAudioFileAsync;
    }

    private async Task<string?> PickAudioFileAsync()
    {
        TopLevel? topLevel = TopLevel.GetTopLevel(this);
        if (topLevel is null)
            return null;

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select Audio",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("Audio")
                {
                    Patterns = new[] { "*.mp3", "*.wav" },
                },
            },
        });

        return files.Count >= 1 ? files[0].Path.LocalPath : null;
    }

    /// <summary>Slider drag start: pause timer-driven position updates.</summary>
    private void Slider_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (DataContext is MainWindowViewModel viewModel)
            viewModel.IsSeeking = true;
    }

    /// <summary>Slider drag end: commit the seek and resume timer-driven updates.</summary>
    private void Slider_PointerReleased(object? sender, PointerReleasedEventArgs e)
    {
        if (DataContext is MainWindowViewModel viewModel)
            viewModel.SeekCommand.Execute(viewModel.TrackPosition);
    }

}