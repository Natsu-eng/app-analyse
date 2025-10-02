# === Script de lancement Streamlit (PowerShell) ===

Write-Host "🚀 Lancement de l'application Streamlit..." -ForegroundColor Green

# Se placer dans la racine du projet
Set-Location "E:\gemini\app-analyse"

# Vérifier si l'environnement virtuel existe
if (Test-Path ".\env\Scripts\Activate.ps1") {
    Write-Host "✅ Activation de l'environnement virtuel..." -ForegroundColor Cyan
    .\env\Scripts\Activate.ps1
} else {
    Write-Host "⚠️ Aucun venv trouvé, lancement sans environnement virtuel." -ForegroundColor Yellow
}

# Lancer Streamlit
streamlit run src/app/main.py

# Garder la console ouverte à la fin
Write-Host "✅ Application arrêtée. Appuyez sur une touche pour fermer." -ForegroundColor Magenta
Pause
