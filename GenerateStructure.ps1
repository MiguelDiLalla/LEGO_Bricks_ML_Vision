# Path to the output file
$outputFile = "FolderStructure.txt"

# Initialize the output content
$output = [System.Collections.Generic.List[string]]::new()

# Function to recursively process the folder structure
function Get-FolderStructure {
    param (
        [string]$folderPath,   # The path of the current folder
        [int]$indentLevel      # Indentation level for readability
    )

    # Generate indentation
    $indent = " " * $indentLevel

    # Add the current folder to the output
    $output.Add("$indent- $(Split-Path $folderPath -Leaf)")

    # Get all files in the current folder
    $files = Get-ChildItem -Path $folderPath -File -ErrorAction SilentlyContinue

    if ($files.Count -gt 20) {
        # Summarize file types if there are more than 20 files
        $output.Add("$indent  More than 20 files, summary:")

        # Count files by extension
        $fileTypeCounts = @{}
        foreach ($file in $files) {
            $ext = $file.Extension.ToLower()
            if (-not $fileTypeCounts.ContainsKey($ext)) {
                $fileTypeCounts[$ext] = 0
            }
            $fileTypeCounts[$ext]++
        }

        # Add summary to the output
        foreach ($ext in $fileTypeCounts.Keys) {
            $output.Add("$indent    ${ext}: $($fileTypeCounts[$ext]) files")
        }
    } else {
        # List all files if there are 20 or fewer, including last modified time
        foreach ($file in $files) {
            $minutesAgo = [math]::Round((New-TimeSpan -Start $file.LastWriteTime -End (Get-Date)).TotalMinutes, 1)
            $output.Add("$indent  $($file.Name) (modified $minutesAgo minutes ago)")
        }
    }

    # Process subdirectories
    $subdirs = Get-ChildItem -Path $folderPath -Directory -ErrorAction SilentlyContinue
    foreach ($subdir in $subdirs) {
        Get-FolderStructure -folderPath $subdir.FullName -indentLevel ($indentLevel + 2)
    }
}

# Start processing from the current directory
Get-FolderStructure -folderPath (Get-Location).Path -indentLevel 0

# Save the output to a file
$output -join "`n" | Set-Content -Path $outputFile -Encoding UTF8

# Inform the user
Write-Output "Folder structure has been saved to '$outputFile'"
