# AutoML X - WorldQuant Foundry Design

A full-stack AI/ML platform with an editorial, modern design inspired by WorldQuant Foundry.

## 🎨 Design Features

- **Mouse-reactive particle vortex** in hero section
- **Moving sparkle particles** with connection effects
- **Monospace typography** for editorial aesthetic
- **Corner-bracket button styling**
- **Split-text layouts** for visual impact
- **Dark theme** with accent colors (#B7FF4A green, cyan blue)
- **Glassmorphic cards** and subtle animations

## 📁 Project Structure

```
automl-x-complete-project/
├── frontend/                    # React Frontend
│   ├── src/
│   │   ├── pages/              # All page components
│   │   │   ├── Landing.js      # Landing page with particle effects
│   │   │   ├── Login.js        # Login page
│   │   │   ├── Signup.js       # Signup page
│   │   │   ├── Dashboard.js    # Main dashboard
│   │   │   ├── Train.js        # Model training page
│   │   │   ├── Experiments.js  # Experiments list
│   │   │   ├── Predict.js      # Prediction page
│   │   │   └── ...
│   │   ├── components/
│   │   │   ├── ui/             # Shadcn UI components
│   │   │   ├── DashboardLayout.js
│   │   │   └── AIChat.js       # AI assistant component
│   │   ├── lib/
│   │   │   ├── api.js          # API client
│   │   │   └── utils.js        # Utility functions
│   │   ├── index.css           # Global styles (WorldQuant theme)
│   │   └── App.js              # Main app component
│   ├── public/
│   ├── package.json
│   └── craco.config.js         # Craco configuration
│
├── backend/                     # FastAPI Backend
│   ├── server.py               # Main FastAPI application
│   ├── models/                 # ML models and utilities
│   ├── utils/                  # AutoML utilities
│   │   ├── automl_brain.py
│   │   ├── feature_engineering.py
│   │   ├── hyperparameter_tuning.py
│   │   └── ...
│   ├── experiments/
│   ├── uploads/
│   ├── requirements.txt
│   └── .env                    # Environment variables
│
├── design_guidelines.md         # Complete design specifications
└── plan.md                     # Development plan & phases
```

## 🚀 Quick Start

### Frontend Setup

1. **Install Dependencies:**
   ```bash
   cd frontend
   yarn install
   ```

2. **Configure Environment:**
   Create `frontend/.env`:
   ```env
   REACT_APP_BACKEND_URL=http://localhost:8001
   ```

3. **Start Development Server:**
   ```bash
   yarn start
   ```

### Backend Setup

1. **Install Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Create `backend/.env`:
   ```env
   MONGO_URL=mongodb://localhost:27017
   DB_NAME=automl_db
   CORS_ORIGINS=*
   SECRET_KEY=your-secret-key-here
   EMERGENT_LLM_KEY=your-llm-key-here  # Optional, for AI chat
   ```

3. **Start Backend Server:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8001 --reload
   ```

## 🎨 Design System

### Color Palette

- **Primary Green:** `#B7FF4A` (Neon lime accent)
- **Secondary Blue:** `#6AA7FF` (Cyan blue)
- **Background Dark:** `#0a0a0a`
- **Surface:** `#111111`
- **Border:** `rgba(255, 255, 255, 0.08)`

### Typography

- **Font Family:** Monospace (JetBrains Mono, Space Mono fallbacks)
- **Headings:** Bold, uppercase, tight tracking
- **Body:** 11-13px, wider tracking for readability

### Components

All UI components are built with **Shadcn/UI** and styled to match the WorldQuant Foundry aesthetic. See `frontend/src/components/ui/` for all available components.

## 🌟 Key Features

### Landing Page
- Mouse-reactive particle vortex (HTML5 Canvas)
- Sparkle particle dots with connection effects
- Split-text hero layout
- Corner-bracket styled CTAs
- Smooth scroll animations

### Dashboard
- Overview metrics (experiments, scores, models)
- Recent experiments table
- Quick action cards
- AI chat assistant
- Clean sidebar navigation

### Train Model
- Drag-and-drop file upload
- Step-by-step workflow indicators
- Real-time training progress
- Results visualization

### Experiments
- Sortable data table
- Experiment filtering
- Model comparison
- Version tracking

## 📦 Dependencies

### Frontend
- React 18
- React Router DOM
- Tailwind CSS
- Framer Motion
- Shadcn UI components
- Lucide React (icons)
- Axios

### Backend
- FastAPI
- Uvicorn
- MongoDB (Motor)
- Scikit-learn
- XGBoost, LightGBM
- Optuna (hyperparameter tuning)
- Pandas, NumPy

## 🔧 Customization

### Merging with Your Backend

1. **Review the structure** in `backend/server.py`
2. **Copy the routes** you need (authentication, experiments, predictions)
3. **Merge the utilities** from `backend/utils/`
4. **Update API endpoints** in `frontend/src/lib/api.js` to match your backend

### Updating Design

- **Colors:** Modify `frontend/src/index.css` global CSS variables
- **Typography:** Update font imports and classes in `index.css`
- **Components:** Customize Shadcn components in `frontend/src/components/ui/`
- **Animations:** Adjust Framer Motion variants in page components

## 📖 Documentation

- **Design Guidelines:** See `design_guidelines.md` for complete design specifications
- **Development Plan:** See `plan.md` for implementation phases and decisions

## 🎯 Production Deployment

### Environment Variables Required

**Frontend:**
- `REACT_APP_BACKEND_URL` - Backend API URL

**Backend:**
- `MONGO_URL` - MongoDB connection string
- `DB_NAME` - Database name
- `SECRET_KEY` - JWT secret key
- `CORS_ORIGINS` - Allowed CORS origins
- `EMERGENT_LLM_KEY` - (Optional) For AI chat features

### Build Frontend

```bash
cd frontend
yarn build
```

The optimized production build will be in `frontend/build/`

## 🤖 AI Assistant Integration

The platform includes an AI chat assistant powered by the Emergent LLM API. To enable:

1. Get your `EMERGENT_LLM_KEY`
2. Add it to `backend/.env`
3. The AI chat button will appear in the dashboard

## 📝 Notes

- The particle effects are optimized for performance (< 60fps)
- All interactive elements have proper `data-testid` attributes for testing
- Mobile-responsive design included
- Dark mode is the default theme

## 🆘 Support

For questions about the design or implementation, refer to:
- `design_guidelines.md` - Complete design system documentation
- `plan.md` - Development decisions and architecture

## 📄 License

This project structure and design implementation is provided as-is for your use.

---

**Built with ❤️ using React, FastAPI, and WorldQuant Foundry-inspired design**
