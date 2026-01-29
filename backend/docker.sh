#!/bin/bash
# ICHeritage Backend - Docker Management Script (Bash)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    echo "ICHeritage Backend Docker Script"
    echo "================================="
    echo "Usage: ./docker.sh <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build backend Docker image"
    echo "  up          Start backend (production mode)"
    echo "  dev         Start backend (development with hot reload)"
    echo "  down        Stop backend"
    echo "  logs        Show backend logs"
    echo "  restart     Restart backend"
    echo "  clean       Remove container and volumes"
    echo "  shell       Open shell in container"
    echo "  test        Run health check test"
    echo "  help        Show this help"
    echo ""
    echo "Examples:"
    echo "  ./docker.sh build"
    echo "  ./docker.sh up"
    echo "  ./docker.sh dev"
}

check_env_file() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${CYAN}📝 Please edit .env and add your GEMINI_API_KEY${NC}"
            exit 1
        else
            echo -e "${RED}Creating minimal .env...${NC}"
            cat > .env << EOF
FLASK_DEBUG=True
PORT=5001
GEMINI_API_KEY=your_api_key_here
EOF
            echo -e "${CYAN}📝 Please edit .env and add your GEMINI_API_KEY${NC}"
            exit 1
        fi
    fi
}

case "${1:-help}" in
    build)
        echo -e "${CYAN}🔨 Building backend image...${NC}"
        docker-compose build backend
        ;;
    up)
        check_env_file
        echo -e "${GREEN}🚀 Starting backend (production)...${NC}"
        docker-compose up -d backend
        echo -e "${GREEN}✅ Backend started at http://localhost:5001${NC}"
        echo "   Health check: http://localhost:5001/api/health"
        ;;
    dev)
        check_env_file
        echo -e "${YELLOW}🔧 Starting backend (development)...${NC}"
        docker-compose --profile dev up -d backend-dev
        echo -e "${GREEN}✅ Backend dev started at http://localhost:5001${NC}"
        echo -e "${CYAN}   Hot reload enabled - changes will auto-reload${NC}"
        ;;
    down)
        echo -e "${YELLOW}🛑 Stopping backend...${NC}"
        docker-compose down
        ;;
    logs)
        docker-compose logs -f
        ;;
    restart)
        echo -e "${CYAN}🔄 Restarting backend...${NC}"
        docker-compose restart
        ;;
    clean)
        echo -e "${RED}🧹 Cleaning up...${NC}"
        docker-compose down -v --remove-orphans
        ;;
    shell)
        container=$(docker-compose ps -q backend 2>/dev/null || docker-compose ps -q backend-dev 2>/dev/null)
        if [ -n "$container" ]; then
            docker exec -it $container /bin/bash
        else
            echo -e "${RED}❌ No running container found${NC}"
        fi
        ;;
    test)
        echo -e "${CYAN}🧪 Testing backend health...${NC}"
        if curl -s http://localhost:5001/api/health > /dev/null; then
            echo -e "${GREEN}✅ Backend is healthy!${NC}"
            curl -s http://localhost:5001/api/health | python -m json.tool
        else
            echo -e "${RED}❌ Backend is not responding${NC}"
        fi
        ;;
    *)
        show_help
        ;;
esac
