# Backend Configuration

The frontend now uses configurable backend URLs that can be easily changed.

## Configuration Options

### 1. Environment Variables (Recommended for Production)

Create a `.env.local` file in the `frontend/` directory:

```bash
# Main backend server for SPHT operations
REACT_APP_SPHT_SERVER=http://localhost:5001

# Upload server for image processing  
REACT_APP_UPLOAD_SERVER=http://localhost:5010
```

### 2. Direct Code Modification

Edit the `BACKEND_CONFIG` object in `src/App.js`:

```javascript
const BACKEND_CONFIG = {
  // Main backend server for SPHT operations
  SPHT_SERVER: 'http://your-backend-url:5001',
  // Upload server for image processing
  UPLOAD_SERVER: 'http://your-backend-url:5010'
};
```

## Usage Examples

### Development with Different Ports
```bash
REACT_APP_SPHT_SERVER=http://localhost:8001
REACT_APP_UPLOAD_SERVER=http://localhost:8010
```

### Production Deployment
```bash
REACT_APP_SPHT_SERVER=https://api.yourstar-detector.com
REACT_APP_UPLOAD_SERVER=https://upload.yourstar-detector.com
```

### Remote Development Server
```bash
REACT_APP_SPHT_SERVER=http://192.168.1.100:5001
REACT_APP_UPLOAD_SERVER=http://192.168.1.100:5010
```

## Default Values

If no environment variables are set, the system defaults to:
- SPHT Server: `http://localhost:5001`
- Upload Server: `http://localhost:5010`

## Important Notes

1. Environment variables must start with `REACT_APP_` to be available in React
2. Restart the development server after changing environment variables
3. The `.env.local` file is gitignored by default (good for security)
4. Environment variables take precedence over hardcoded values 