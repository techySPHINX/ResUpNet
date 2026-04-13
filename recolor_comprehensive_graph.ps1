Add-Type -AssemblyName System.Drawing

$inputPath = "model_comparison_comprehensive.png"
$outputPath = "model_comparison_comprehensive_recolored.png"

if (-not (Test-Path $inputPath)) {
    throw "Input image not found: $inputPath"
}

function Clamp-Byte([int]$value) {
    if ($value -lt 0) { return 0 }
    if ($value -gt 255) { return 255 }
    return $value
}

# Original colors in the current graph (anchors)
$colorMap = @(
    @{ Name = "ResNet";       Old = @(255, 107, 107); New = @(31, 119, 180) },
    @{ Name = "UNet";         Old = @(78, 205, 196);  New = @(255, 127, 14) },
    @{ Name = "AttentionUNet";Old = @(149, 225, 211); New = @(44, 160, 44) },
    @{ Name = "ResUpNet";     Old = @(240, 128, 128); New = @(214, 39, 40) }
)

# Thresholds chosen to recolor only model strokes/legend lines while preserving background/text
$saturationThreshold = 18
$distanceThresholdSq = 3600  # RGB distance <= 60

$bmp = [System.Drawing.Bitmap]::new($inputPath)

try {
    $width = $bmp.Width
    $height = $bmp.Height

    Write-Host "Processing image: ${width}x${height}"

    for ($y = 0; $y -lt $height; $y++) {
        for ($x = 0; $x -lt $width; $x++) {
            $px = $bmp.GetPixel($x, $y)

            $maxC = [Math]::Max($px.R, [Math]::Max($px.G, $px.B))
            $minC = [Math]::Min($px.R, [Math]::Min($px.G, $px.B))
            $sat = $maxC - $minC

            if ($sat -le $saturationThreshold) {
                continue
            }

            $bestIndex = -1
            $bestDist = [int]::MaxValue

            for ($i = 0; $i -lt $colorMap.Count; $i++) {
                $old = $colorMap[$i].Old
                $dr = $px.R - $old[0]
                $dg = $px.G - $old[1]
                $db = $px.B - $old[2]
                $distSq = ($dr * $dr) + ($dg * $dg) + ($db * $db)

                if ($distSq -lt $bestDist) {
                    $bestDist = $distSq
                    $bestIndex = $i
                }
            }

            if ($bestDist -lt $distanceThresholdSq) {
                $old = $colorMap[$bestIndex].Old
                $new = $colorMap[$bestIndex].New

                # Preserve anti-aliasing/light-dark edge detail by transferring offset
                $newR = Clamp-Byte ($new[0] + ($px.R - $old[0]))
                $newG = Clamp-Byte ($new[1] + ($px.G - $old[1]))
                $newB = Clamp-Byte ($new[2] + ($px.B - $old[2]))

                $bmp.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($newR, $newG, $newB))
            }
        }

        if (($y + 1) % 100 -eq 0) {
            Write-Host "Processed rows: $($y + 1) / $height"
        }
    }

    $bmp.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Host "Saved recolored graph: $outputPath"
}
finally {
    $bmp.Dispose()
}
