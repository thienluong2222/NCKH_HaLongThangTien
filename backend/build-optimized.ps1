# Build optimized Docker image
# Expected: ~3-4GB instead of 13.9GB

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building OPTIMIZED Backend Docker Image" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Build the optimized image
Write-Host "Building image with CPU-only PyTorch..." -ForegroundColor Yellow
docker build -f Dockerfile.optimized -t icheritage-backend:optimized .

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Build Complete! Comparing sizes:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Show image sizes
docker images | Select-String "icheritage-backend|REPOSITORY"

Write-Host ""
Write-Host "To run the optimized image:" -ForegroundColor Cyan
Write-Host "  docker run -p 5001:5001 -v `${PWD}/weight:/app/weight icheritage-backend:optimized" -ForegroundColor White
