# Cybersecurity Analytic System

A comprehensive cybersecurity risk assessment system that combines machine learning models with LLM-powered expert analysis to provide real-time security insights and educational feedback.

## System Architecture

This system integrates three specialized deep learning models with ChatGPT-powered analysis:
- **URL Risk Analysis**: Detects malicious URLs using entropy, domain features, and structural patterns
- **Network Traffic Analysis**: Identifies suspicious network flows using packet statistics and traffic patterns  
- **User Behavior Analysis**: Monitors login patterns, session behavior, and access anomalies
- **LLM Expert Analysis**: Provides detailed explanations and security insights using ChatGPT

The system provides an interactive teaching interface where users can assess security scenarios and compare their judgment with AI predictions, enhanced by professional LLM-generated analysis.

## Key Features

- **Real-time Risk Assessment**: Instant analysis of URL, network, and user data using deep learning models
- **Dynamic Risk Weighting**: Advanced algorithms that amplify confidence when multiple high-risk indicators are detected
- **LLM Expert Insights**: Professional analysis with markdown formatting and keyword highlighting
- **Interactive Learning Interface**: Flip-card UI for user assessment and AI comparison
- **Intelligent Feature Highlighting**: Visual indicators showing which specific features contribute to risk based on deep learning predictions
- **Loading Experience**: Smooth animations and status updates during processing
- **Comprehensive Data Support**: Handles missing values and provides fallback mechanisms

## Project Structure

```
cybersec_fusion_system/
├── backend/                    # Python Flask API server
│   ├── app.py                 # Main Flask application with prediction endpoints
│   ├── models/                # Trained DL models (.h5) and scalers (.pkl)
│   ├── features/              # Feature definition files (.txt)
│   ├── data/                  # Training and sample datasets (.csv)
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React TypeScript web application
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── FlipAssessmentCard.tsx  # Main assessment interface
│   │   │   └── SampleViewer.tsx        # Feature display component
│   │   ├── services/          # API communication layer
│   │   └── types.ts          # TypeScript type definitions
│   └── public/               # Static assets
└── scripts/                  # Utility and test scripts
```

## Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Quick Start Script
```bash
chmod +x start_backend.sh
./start_backend.sh
```

## Usage

1. **Start Services**: Launch both backend (port 5000) and frontend (port 3000)
2. **Load Sample**: System automatically fetches a random sample on startup
3. **Assess Risk**: Review the displayed features and make your security assessment
4. **Compare Results**: Submit your decision to see AI prediction and expert analysis
5. **New Assessment**: Click "New Assessment" to analyze another sample

## Deep Learning Models

### URL Model
- **Architecture**: 1D CNN with feature normalization
- **Features**: URL entropy, character counts, structural patterns, domain reputation
- **Output**: Risk score (0-1) where 1 indicates malicious content

### Network Model  
- **Architecture**: Deep neural network with traffic flow analysis
- **Features**: Packet statistics, flow duration, data rates, protocol patterns
- **Output**: Risk score (0-1) for network traffic anomalies

### User Model
- **Architecture**: Behavioral pattern classifier
- **Features**: Login patterns, session metrics, access timing, IP reputation
- **Output**: Risk score (0-1) for user behavior anomalies

## LLM Integration

### Expert Analysis Features
- **Real-time Analysis**: ChatGPT provides instant security insights
- **Markdown Formatting**: Professional formatting with bold, code, and blockquote styling
- **Keyword Highlighting**: Important terms are visually emphasized
- **Educational Focus**: Explanations designed for cybersecurity learning

### Analysis Components
- **Risk Assessment Explanation**: Why the system gave specific results
- **Feature Contribution Analysis**: Which specific features contributed most to risk
- **Pattern Comparison**: Normal vs abnormal patterns for concerning features
- **Security Implications**: Brief explanation of potential threats and recommendations

## Dynamic Risk Logic

The system implements sophisticated risk weighting:
- **Individual Thresholds**: 90%+ (extreme), 70%+ (high), 50%+ (elevated)
- **Cumulative Boosting**: Multiple risk factors increase overall confidence
- **Confidence Caps**: Maximum 95% confidence with safeguards against false positives
- **Fallback Mechanisms**: Graceful handling of missing data and model failures

## User Interface

- **Clean Design**: Modern, medical-report-inspired visual style
- **Feature Highlighting**: Color-coded indicators (red=danger, orange=abnormal)
- **Flip Animation**: Smooth transition between assessment and results
- **Responsive Layout**: Works across desktop and mobile devices
- **Accessibility**: High contrast colors and clear typography
- **Loading Experience**: Professional animations during processing

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /sample` - Fetch random sample data
- `POST /predict` - Submit sample for risk analysis with LLM insights

## Data Processing

The system handles:
- **Missing Values**: Intelligent imputation and default handling
- **Data Normalization**: Standardized feature scaling across models
- **Feature Engineering**: Automated extraction of security-relevant patterns
- **Real-time Processing**: Sub-second response times for all deep learning predictions
- **LLM Integration**: Seamless ChatGPT API integration for expert analysis

## Deployment

### Development Mode
- Backend: `python app.py` (Flask dev server)
- Frontend: `npm start` (React dev server)

### Production Mode
- Backend: Use gunicorn or uwsgi with proper WSGI configuration
- Frontend: `npm run build` and serve with nginx or similar

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Submit a pull request with detailed description

## License

This project is developed for educational and research purposes in cybersecurity.

## Troubleshooting

### Common Issues
- **Port Conflicts**: Ensure ports 3000 and 5000 are available
- **Model Loading**: Verify all `.h5` and `.pkl` files are present in `/models/`
- **Dependencies**: Run `pip install -r requirements.txt` and `npm install`
- **Data Files**: Ensure CSV datasets are present in `/data/` directory
- **LLM API**: Verify OpenAI API key is configured in backend

### LLM Configuration
To enable ChatGPT expert analysis, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Debug Mode
Set `DEBUG=True` in `app.py` for detailed error logging and hot reloading.

