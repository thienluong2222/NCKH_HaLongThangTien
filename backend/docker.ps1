# ICHeritage Backend - Docker Management Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host "ICHeritage Backend Docker Script"
    Write-Host "================================="
    Write-Host "Usage: .\docker.ps1 <command>"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  build       Build backend Docker image"
    Write-Host "  up          Start backend (production mode)"
    Write-Host "  dev         Start backend (development with hot reload)"
    Write-Host "  down        Stop backend"
    Write-Host "  logs        Show backend logs"
    Write-Host "  restart     Restart backend"
    Write-Host "  clean       Remove container and volumes"
    Write-Host "  shell       Open shell in container"
    Write-Host "  test        Run health check test"
    Write-Host "  help        Show this help"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\docker.ps1 build"
    Write-Host "  .\docker.ps1 up"
    Write-Host "  .\docker.ps1 dev"
    Write-Host "  .\docker.ps1 logs"
}

function Check-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Host "[WARNING] .env file not found. Creating from .env.example..." -ForegroundColor Yellow
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Host "[INFO] Please edit .env and add your GEMINI_API_KEY" -ForegroundColor Cyan
            exit 1
        } else {
            Write-Host "[ERROR] .env.example not found! Creating minimal .env..." -ForegroundColor Red
            $envContent = "FLASK_DEBUG=True`nPORT=5001`nGEMINI_API_KEY=your_api_key_here"
            $envContent | Out-File -FilePath ".env" -Encoding utf8
            Write-Host "[INFO] Please edit .env and add your GEMINI_API_KEY" -ForegroundColor Cyan
            exit 1
        }
    }
}

switch ($Command.ToLower()) {
    "build" {
        Write-Host "[BUILD] Building backend image..." -ForegroundColor Cyan
        docker-compose build backend
    }
    "up" {
        Check-EnvFile
        Write-Host "[START] Starting backend (production)..." -ForegroundColor Green
        docker-compose up -d backend
        Write-Host "[OK] Backend started at http://localhost:5001" -ForegroundColor Green
        Write-Host "     Health check: http://localhost:5001/api/health" -ForegroundColor White
    }
    "dev" {
        Check-EnvFile
        Write-Host "[DEV] Starting backend (development)..." -ForegroundColor Yellow
        docker-compose --profile dev up -d backend-dev
        Write-Host "[OK] Backend dev started at http://localhost:5001" -ForegroundColor Green
        Write-Host "     Hot reload enabled - changes will auto-reload" -ForegroundColor Cyan
    }
    "down" {
        Write-Host "[STOP] Stopping backend..." -ForegroundColor Yellow
        docker-compose down
    }
    "logs" {
        docker-compose logs -f
    }
    "restart" {
        Write-Host "[RESTART] Restarting backend..." -ForegroundColor Cyan
        docker-compose restart
    }
    "clean" {
        Write-Host "[CLEAN] Cleaning up..." -ForegroundColor Red
        docker-compose down -v --remove-orphans
    }
    "shell" {
        $container = docker-compose ps -q backend
        if (-not $container) {
            $container = docker-compose ps -q backend-dev
        }
        if ($container) {
            docker exec -it $container /bin/bash
        } else {
            Write-Host "[ERROR] No running container found" -ForegroundColor Red
        }
    }
    "test" {
        Write-Host "[TEST] Testing backend health..." -ForegroundColor Cyan
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:5001/api/health" -Method Get
            Write-Host "[OK] Backend is healthy!" -ForegroundColor Green
            $response | ConvertTo-Json
        } catch {
            Write-Host "[ERROR] Backend is not responding" -ForegroundColor Red
            Write-Host $_.Exception.Message
        }
    }
    default {
        Show-Help
    }
}
