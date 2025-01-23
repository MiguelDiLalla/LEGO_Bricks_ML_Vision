# Define el archivo de salida
$output_file = "estructura.txt"

# Obtén el directorio actual
$current_dir = Get-Location

# Generar la estructura
Get-ChildItem -Recurse -Path $current_dir | Out-File -FilePath $output_file
Write-Host "Estructura generada exitosamente en: $output_file"
