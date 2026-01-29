# ICHeritage Docker Management Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host @"
ICHeritage Docker Management Script
====================================
Usage: .\docker-manage.ps1 <command>

Commands:
  build       Build all Docker images
  up          Start all services (production)
  up-dev      Start all services (development with hot reload)
  down        Stop all services
  logs        Show logs from all services
  logs-be     Show backend logs only
  logs-fe     Show frontend logs only
  restart     Restart all services
  clean       Stop and remove all containers, networks, volumes
  status      Show status of all services
  shell-be    Open shell in backend container
  help        Show this help message

Examples:
  .\docker-manage.ps1 build
  .\docker-manage.ps1 up
  .\docker-manage.ps1 logs-be
"@
}

function Check-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Host "⚠️  .env file not found. Creating from .env.docker.example..." -ForegroundColor Yellow
        if (Test-Path ".env.docker.example") {
            Copy-Item ".env.docker.example" ".env"
            Write-Host "📝 Please edit .env file and add your GEMINI_API_KEY" -ForegroundColor Cyan
            exit 1
        } else {
            Write-Host "❌ .env.docker.example not found!" -ForegroundColor Red
            exit 1
        }
    }
}

switch ($Command.ToLower()) {
    "build" {
        Write-Host "🔨 Building Docker images..." -ForegroundColor Cyan
        docker-compose build
    }
    "up" {
        Check-EnvFile
        Write-Host "🚀 Starting production services..." -ForegroundColor Green
        docker-compose up -d
        Write-Host "✅ Services started!" -ForegroundColor Green
        Write-Host "   Frontend: http://localhost:80" -ForegroundColor White
        Write-Host "   Backend:  http://localhost:5001" -ForegroundColor White
    }
    "up-dev" {
        Check-EnvFile
        Write-Host "🔧 Starting development services..." -ForegroundColor Yellow
        docker-compose -f docker-compose.dev.yml up -d
        Write-Host "✅ Dev services started!" -ForegroundColor Green
    }
    "down" {
        Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
        docker-compose down
        docker-compose -f docker-compose.dev.yml down 2>$null
    }
    "logs" {
        docker-compose logs -f
    }
    "logs-be" {
        docker-compose logs -f backend
    }
    "logs-fe" {
        docker-compose logs -f frontend
    }
    "restart" {
        Write-Host "🔄 Restarting services..." -ForegroundColor Cyan
        docker-compose restart
    }
    "clean" {
        Write-Host "🧹 Cleaning up..." -ForegroundColor Red
        docker-compose down -v --remove-orphans
        docker-compose -f docker-compose.dev.yml down -v --remove-orphans 2>$null
    }
    "status" {
        docker-compose ps
    }
    "shell-be" {
        docker-compose exec backend /bin/bash
    }
    default {
        Show-Help
    }
}
