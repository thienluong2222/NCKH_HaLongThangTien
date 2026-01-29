#!/bin/bash
# ICHeritage Docker Management Script (Bash)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo "ICHeritage Docker Management Script"
    echo "===================================="
    echo "Usage: ./docker-manage.sh <command>"
    echo ""
    echo "Commands:"
    echo "  build       Build all Docker images"
    echo "  up          Start all services (production)"
    echo "  up-dev      Start all services (development with hot reload)"
    echo "  down        Stop all services"
    echo "  logs        Show logs from all services"
    echo "  logs-be     Show backend logs only"
    echo "  logs-fe     Show frontend logs only"
    echo "  restart     Restart all services"
    echo "  clean       Stop and remove all containers, networks, volumes"
    echo "  status      Show status of all services"
    echo "  shell-be    Open shell in backend container"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./docker-manage.sh build"
    echo "  ./docker-manage.sh up"
    echo "  ./docker-manage.sh logs-be"
}

check_env_file() {
    if [ ! -f ".env" ]; then
        echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.docker.example...${NC}"
        if [ -f ".env.docker.example" ]; then
            cp .env.docker.example .env
            echo -e "${CYAN}📝 Please edit .env file and add your GEMINI_API_KEY${NC}"
            exit 1
        else
            echo -e "${RED}❌ .env.docker.example not found!${NC}"
            exit 1
        fi
    fi
}

case "${1:-help}" in
    build)
        echo -e "${CYAN}🔨 Building Docker images...${NC}"
        docker-compose build
        ;;
    up)
        check_env_file
        echo -e "${GREEN}🚀 Starting production services...${NC}"
        docker-compose up -d
        echo -e "${GREEN}✅ Services started!${NC}"
        echo "   Frontend: http://localhost:80"
        echo "   Backend:  http://localhost:5001"
        ;;
    up-dev)
        check_env_file
        echo -e "${YELLOW}🔧 Starting development services...${NC}"
        docker-compose -f docker-compose.dev.yml up -d
        echo -e "${GREEN}✅ Dev services started!${NC}"
        ;;
    down)
        echo -e "${YELLOW}🛑 Stopping services...${NC}"
        docker-compose down
        docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
        ;;
    logs)
        docker-compose logs -f
        ;;
    logs-be)
        docker-compose logs -f backend
        ;;
    logs-fe)
        docker-compose logs -f frontend
        ;;
    restart)
        echo -e "${CYAN}🔄 Restarting services...${NC}"
        docker-compose restart
        ;;
    clean)
        echo -e "${RED}🧹 Cleaning up...${NC}"
        docker-compose down -v --remove-orphans
        docker-compose -f docker-compose.dev.yml down -v --remove-orphans 2>/dev/null || true
        ;;
    status)
        docker-compose ps
        ;;
    shell-be)
        docker-compose exec backend /bin/bash
        ;;
    *)
        show_help
        ;;
esac
