# Festival Detection Angular App

Modern Angular 18+ frontend for the Vietnamese Cultural Festival Detection System.

## Features

- 🎨 **Angular Material UI** - Modern, responsive design
- ⚡ **Standalone Components** - Latest Angular architecture  
- 📊 **Reactive State** - Using Angular Signals
- 🎬 **Drag & Drop Upload** - Images and videos
- 💬 **AI Chat Interface** - Natural language interaction
- 📈 **Plotly Charts** - Interactive visualizations
- ✨ **Smooth Animations** - Enhanced UX

## Quick Start

```bash
# Navigate to the angular directory
cd interface-angular

# Install dependencies
npm install

# Start development server
ng serve

# Open http://localhost:4200
```

## Project Structure

```
src/app/
├── components/
│   ├── header/              # App header with branding
│   ├── upload/              # Drag-drop file upload
│   ├── chat-box/            # AI chat interface
│   ├── results-card/        # Winner display with animations
│   ├── explanation-tabs/    # Charts and rule explanations
│   └── footer/              # App footer
├── services/
│   ├── festival-detection.service.ts  # Main detection logic
│   └── chat.service.ts                # Chat state management
└── app.component.ts         # Root component
```

## Backend Integration

Update the API URL in `festival-detection.service.ts`:

```typescript
private readonly API_URL = 'http://localhost:8000/api';
```

## Build for Production

```bash
ng build --configuration production
```

Output will be in `dist/festival-detection/`.
