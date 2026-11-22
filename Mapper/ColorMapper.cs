#nullable enable

using System;
using Avalonia.Media;

/// <summary>
/// Maps valence/arousal to a colour using Russell's circumplex model: four anchor
/// colours sit at the corners of the 2D mood space and are bilinearly interpolated,
/// so mid-range values stay pleasant instead of drifting through muddy hues.
/// </summary>
public static class ColorMapper
{
    // Arousal = 1 (top edge).
    private const string TopLeftHex = "#E53935";    // valence 0, arousal 1 - anger / fear
    private const string TopRightHex = "#FFCA28";   // valence 1, arousal 1 - joy
    private const string BottomLeftHex = "#1E3A8A"; // valence 0, arousal 0 - sadness
    private const string BottomRightHex = "#26A69A";// valence 1, arousal 0 - calm

    private static readonly Color TopLeft = Color.Parse(TopLeftHex);
    private static readonly Color TopRight = Color.Parse(TopRightHex);
    private static readonly Color BottomLeft = Color.Parse(BottomLeftHex);
    private static readonly Color BottomRight = Color.Parse(BottomRightHex);

    /// <summary>
    /// Bilinear interpolation of the four circumplex anchors.
    /// Valence moves along the horizontal axis, arousal along the vertical axis.
    /// </summary>
    public static Color MapToColor(float valence, float arousal)
    {
        double v = Math.Clamp(valence, 0f, 1f);
        double a = Math.Clamp(arousal, 0f, 1f);

        // Bottom edge (arousal 0) and top edge (arousal 1), interpolated by valence.
        Color bottom = Blend(BottomLeft, BottomRight, v);
        Color top = Blend(TopLeft, TopRight, v);

        // Finally blend the two edges by arousal.
        return Blend(bottom, top, a);
    }

    private static Color Blend(Color from, Color to, double amount)
    {
        double t = Math.Clamp(amount, 0.0, 1.0);

        return Color.FromRgb(
            Lerp(from.R, to.R, t),
            Lerp(from.G, to.G, t),
            Lerp(from.B, to.B, t));
    }

    private static byte Lerp(byte from, byte to, double amount)
        => (byte)Math.Clamp(
            Math.Round(from + (to - from) * amount, MidpointRounding.AwayFromZero),
            0.0,
            255.0);
}