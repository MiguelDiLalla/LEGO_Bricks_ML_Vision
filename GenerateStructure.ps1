# Define el archivo de salida
$output_file = "estructura.txt"

# Obtén el directorio actual
$current_dir = Get-Location

# Verifica si 'src' está en el directorio actual
if (Test-Path "$current_dir\src") {
    $target_dir = $current_dir
    Write-Host "Directorio 'src' encontrado en el directorio actual: $target_dir"
}
# Verifica si 'src' está en el directorio padre
elseif (Test-Path "$current_dir\..\\src") {
    $target_dir = Resolve-Path "$current_dir\.."
    Write-Host "Directorio 'src' encontrado como carpeta hermana. Generando estructura en: $target_dir"
}
else {
    Write-Host "No se encontró la carpeta 'src'. No se generará el archivo de estructura."
    return
}

# Generar la estructura
Get-ChildItem -Recurse -Path $target_dir | Out-File -FilePath $output_file
Write-Host "Estructura generada exitosamente en: $output_file"
