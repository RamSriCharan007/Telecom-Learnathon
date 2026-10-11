Get-ChildItem -Path 'd:\l\.venv' -Recurse -File | ForEach-Object { Unblock-File -Path $_.FullName }
Write-Host "Unblocked all files in .venv!"
