#!/bin/bash

# AutoML X - Quick Setup Script
# This script helps you quickly set up the project

echo "🚀 AutoML X - Quick Setup"
echo "=========================="
echo ""

# Check if we're in the right directory
if [ ! -d "frontend" ] || [ ! -d "backend" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   (where frontend/ and backend/ folders are located)"
    exit 1
fi

echo "📦 Step 1: Setting up Frontend..."
cd frontend

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating frontend/.env from example..."
    cp .env.example .env
    echo "✅ Frontend .env created"
else
    echo "⚠️  Frontend .env already exists, skipping..."
fi

# Install frontend dependencies
echo "Installing frontend dependencies..."
if command -v yarn &> /dev/null; then
    yarn install
    echo "✅ Frontend dependencies installed"
else
    echo "❌ Yarn not found. Please install Yarn: npm install -g yarn"
    exit 1
fi

cd ..

echo ""
echo "🔧 Step 2: Setting up Backend..."
cd backend

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating backend/.env from example..."
    cp .env.example .env
    echo "✅ Backend .env created"
    echo "⚠️  IMPORTANT: Edit backend/.env and set your SECRET_KEY and database credentials"
else
    echo "⚠️  Backend .env already exists, skipping..."
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo "Installing backend dependencies..."
source venv/bin/activate
pip install -r requirements.txt
echo "✅ Backend dependencies installed"

cd ..

echo ""
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Configure Environment Variables:"
echo "   - Edit backend/.env with your database credentials"
echo "   - Generate a secure SECRET_KEY (use: python -c 'import secrets; print(secrets.token_urlsafe(32))')"
echo "   - Update REACT_APP_BACKEND_URL in frontend/.env if needed"
echo ""
echo "2. Start MongoDB:"
echo "   mongod --dbpath /path/to/your/data/directory"
echo ""
echo "3. Start Backend:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   uvicorn server:app --host 0.0.0.0 --port 8001 --reload"
echo ""
echo "4. Start Frontend (in a new terminal):"
echo "   cd frontend"
echo "   yarn start"
echo ""
echo "5. Open your browser:"
echo "   http://localhost:3000"
echo ""
echo "🎉 Happy coding!"
