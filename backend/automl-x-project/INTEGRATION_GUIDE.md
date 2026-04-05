# 🔄 Integration Guide - Merging with Your Existing Backend

This guide will help you integrate the AutoML X WorldQuant Foundry design into your existing project.

## 📋 What's Included in This Package

```
automl-x-project/
├── README.md                    # Main documentation
├── INTEGRATION_GUIDE.md         # This file
├── setup.sh                     # Quick setup script
├── design_guidelines.md         # Complete design system
├── plan.md                      # Development decisions
├── frontend/                    # Complete React frontend
└── backend/                     # Reference FastAPI backend
```

## 🎯 Integration Strategies

### Option 1: Frontend Only (Recommended)

If you already have a working backend and just want the UI:

1. **Copy the frontend folder** to your project
2. **Update API endpoints** in `frontend/src/lib/api.js` to match your backend
3. **Configure environment** in `frontend/.env`
4. **Install and run**:
   ```bash
   cd frontend
   yarn install
   yarn start
   ```

### Option 2: Backend Reference

If you want to see how certain features work:

1. **Review backend structure** in `backend/`
2. **Copy specific routes** you need (auth, experiments, etc.)
3. **Adapt to your database** (currently uses MongoDB)
4. **Merge utility functions** from `backend/utils/`

### Option 3: Complete Integration

For merging everything into your existing project:

1. Use an AI agent (like the one that helped you create this)
2. Provide both codebases
3. Ask it to merge frontend design with your backend logic
4. Review and test the merged code

## 🔧 Frontend Integration Steps

### Step 1: Copy Files

```bash
# Copy entire frontend to your project
cp -r automl-x-project/frontend /path/to/your/project/

# Or copy specific parts:
cp -r automl-x-project/frontend/src/pages /path/to/your/project/frontend/src/
cp -r automl-x-project/frontend/src/components /path/to/your/project/frontend/src/
cp automl-x-project/frontend/src/index.css /path/to/your/project/frontend/src/
```

### Step 2: Update API Configuration

Edit `frontend/src/lib/api.js`:

```javascript
import axios from 'axios';

const API = axios.create({
  baseURL: process.env.REACT_APP_BACKEND_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Update interceptors to match your auth system
API.interceptors.request.use((config) => {
  const token = localStorage.getItem('token'); // Or your token storage
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default API;
```

### Step 3: Map API Endpoints

Update all API calls to match your backend routes:

**Current structure** (in Landing.js, Dashboard.js, etc.):
```javascript
// Login
await API.post('/api/login', { email, password })

// Get experiments
await API.get('/api/experiments')

// Train model
await API.post('/api/train', formData)
```

**Map to your endpoints:**
```javascript
// Example: If your endpoints are different
await API.post('/auth/login', { email, password })
await API.get('/experiments/list')
await API.post('/ml/train', formData)
```

### Step 4: Install Dependencies

The frontend uses these key dependencies:

```json
{
  "react": "^18.2.0",
  "react-router-dom": "^6.x",
  "framer-motion": "^11.x",
  "tailwindcss": "^3.x",
  "lucide-react": "^0.x",
  "axios": "^1.x"
}
```

Install them:
```bash
cd frontend
yarn install
```

### Step 5: Configure Tailwind

Ensure your `tailwind.config.js` includes:

```javascript
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      // See design_guidelines.md for complete config
    },
  },
}
```

## 🎨 Design System Integration

### Colors

The theme uses these primary colors (defined in `index.css`):

```css
:root {
  --primary-green: #B7FF4A;
  --secondary-blue: #6AA7FF;
  --background-dark: #0a0a0a;
  --surface: #111111;
}
```

### Typography

Monospace fonts are key to the design:
- Primary: JetBrains Mono
- Fallbacks: Space Mono, Courier New

### Components

All UI components are in `frontend/src/components/ui/` (Shadcn-based).

## 🔄 Backend Integration (If Needed)

### Database Models

If merging backend logic, map these MongoDB models to your database:

**Users:**
```python
{
  "user_id": "uuid",
  "email": "string",
  "hashed_password": "string",
  "created_at": "datetime"
}
```

**Experiments:**
```python
{
  "experiment_id": "uuid",
  "user_id": "uuid",
  "model_type": "string",
  "score": "float",
  "created_at": "datetime"
}
```

### API Routes

Key routes in the reference backend:

- `POST /api/login` - User authentication
- `POST /api/signup` - User registration
- `GET /api/experiments` - List all experiments
- `POST /api/train` - Train new model
- `POST /api/predict` - Make predictions
- `GET /api/leaderboard` - Get leaderboard

Adapt these to your existing API structure.

## 🧪 Testing the Integration

### 1. Start Your Backend
```bash
# Your existing backend
cd /path/to/your/backend
# Start it however you normally do
```

### 2. Configure Frontend
```bash
# Update frontend/.env
REACT_APP_BACKEND_URL=http://localhost:YOUR_PORT
```

### 3. Start Frontend
```bash
cd frontend
yarn start
```

### 4. Test Key Flows
- ✅ Landing page loads with particle effects
- ✅ Login/Signup works with your auth
- ✅ Dashboard shows data from your backend
- ✅ All interactive elements work

## 🎭 Customizing the Design

### Change Colors

Edit `frontend/src/index.css`:

```css
:root {
  --primary-green: #YOUR_COLOR;
  --secondary-blue: #YOUR_COLOR;
}
```

### Modify Particle Effects

Edit `frontend/src/pages/Landing.js`:

```javascript
// Line ~39: Particle count
const NUM = 700; // Decrease for better performance

// Line ~76-83: Mouse interaction strength
const force = (1 - dist / maxDist) * 60; // Adjust multiplier
```

### Update Typography

Edit `frontend/src/index.css`:

```css
body {
  font-family: 'Your Font', monospace;
}
```

## 🚀 Production Deployment

### Build Frontend

```bash
cd frontend
yarn build
```

### Serve Static Files

The `build/` folder contains optimized production files.

Serve with:
- Nginx
- Apache
- Your existing web server
- Vercel/Netlify (automatic)

### Environment Variables

Production `.env`:

```env
REACT_APP_BACKEND_URL=https://your-api.com
```

## 📦 Using with AI Agents

To merge this with your existing project using an AI agent:

1. **Provide both codebases:**
   - This zip file (AutoML X design)
   - Your existing backend code

2. **Clear instructions:**
   ```
   "Merge the AutoML X frontend design into my existing project.
   Keep my backend logic intact and update the frontend API calls
   to work with my endpoints. Maintain the WorldQuant Foundry
   design aesthetic."
   ```

3. **Specify what to keep:**
   - Your: Database, auth logic, business logic, API structure
   - This: UI components, design system, particle effects, layouts

4. **Test thoroughly** after merging

## 🆘 Common Issues

### Issue: Particle effects lag
**Solution:** Reduce particle count in Landing.js (line 39)

### Issue: API calls fail
**Solution:** Check CORS settings and API base URL

### Issue: Styles not applying
**Solution:** Ensure Tailwind config includes all content paths

### Issue: Build fails
**Solution:** Check Node version (requires 16+) and reinstall dependencies

## 📚 Additional Resources

- `design_guidelines.md` - Complete design specifications
- `README.md` - Quick start guide
- `plan.md` - Development decisions and architecture
- `frontend/src/pages/Landing.js` - Particle effects implementation
- `frontend/src/index.css` - Global styles and theme

## 🤝 Need Help?

If you encounter issues during integration:

1. Check the design_guidelines.md for design specifications
2. Review the reference backend code for API structure
3. Use an AI agent to help with complex merges
4. Test incrementally (one page/feature at a time)

---

**Happy integrating! 🎉**
