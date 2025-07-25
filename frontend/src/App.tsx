import React, { useState, useEffect } from 'react';
import './App.css';
import { SampleData, UserDecision as UserDecisionType, PredictionResponse } from './types';
import { api } from './services/api';
import SampleViewer from './components/SampleViewer';
import UserDecision from './components/UserDecision';
import RiskResult from './components/RiskResult';

function App() {
  const [sample, setSample] = useState<SampleData | null>(null);
  const [urlSampleId, setUrlSampleId] = useState<number | null>(null);
  const [networkSampleId, setNetworkSampleId] = useState<number | null>(null);
  const [userSampleId, setUserSampleId] = useState<number | null>(null);
  const [userDecision, setUserDecision] = useState<UserDecisionType | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking');

  // Check backend connection on mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      await api.healthCheck();
      setBackendStatus('connected');
    } catch (err) {
      setBackendStatus('error');
      setError('Unable to connect to backend server, please ensure backend is running');
    }
  };

  const getNewSample = async () => {
    setIsLoading(true);
    setError(null);
    setSample(null);
    setUrlSampleId(null);
    setNetworkSampleId(null);
    setUserSampleId(null);
    setUserDecision(null);
    setPrediction(null);

    try {
      const response = await api.getRandomSample();
      setSample(response.sample);
      setUrlSampleId(response.url_sample_id);
      setNetworkSampleId(response.network_sample_id);
      setUserSampleId(response.user_sample_id);
    } catch (err) {
      setError('Failed to fetch sample, please try again');
      console.error('Error fetching sample:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!sample || !userDecision || urlSampleId === null || networkSampleId === null || userSampleId === null) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const response = await api.predictSample(sample, urlSampleId, networkSampleId, userSampleId);
      setPrediction(response);
    } catch (err) {
      setError('Prediction failed, please try again');
      console.error('Error predicting:', err);
    } finally {
      setIsSubmitting(false);
    }
  };



  if (backendStatus === 'checking') {
    return (
      <div style={{ 
        minHeight: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <div className="text-center">
          <div style={{
            width: '64px',
            height: '64px',
            border: '4px solid #3b82f6',
            borderTop: '4px solid transparent',
            borderRadius: '50%',
            margin: '0 auto 24px',
            animation: 'spin 1s linear infinite'
          }}></div>
          <p style={{ color: '#4b5563', fontSize: '18px', fontWeight: '500' }}>
            Checking backend connection...
          </p>
        </div>
      </div>
    );
  }

  if (backendStatus === 'error') {
    return (
      <div style={{ 
        minHeight: '100vh', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        <div className="card p-8 text-center" style={{ maxWidth: '500px' }}>
          <div style={{ fontSize: '80px', marginBottom: '24px' }}>⚠️</div>
          <h1 className="text-3xl font-bold text-gray-800 mb-4">Connection Error</h1>
          <p className="text-gray-600 mb-8" style={{ fontSize: '18px', marginBottom: '32px' }}>
            {error}
          </p>
          <button
            onClick={checkBackendConnection}
            className="btn btn-primary"
            style={{ padding: '12px 32px', fontSize: '16px' }}
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Header */}
      <header className="bg-white sticky top-0 z-50" style={{
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
        borderBottom: '1px solid rgba(229, 231, 235, 0.5)'
      }}>
        <div className="container py-6">
          <div className="flex justify-between items-center" style={{ gap: '24px' }}>
            <div style={{ flex: 1 }}>
              <h1 className="text-4xl font-bold gradient-text">
                 Cybersecurity Analytic System
              </h1>
            </div>
            <div className="flex items-center" style={{ gap: '16px' }}>
              <div className="flex items-center" style={{
                color: '#059669',
                backgroundColor: '#ecfdf5',
                padding: '8px 16px',
                borderRadius: '9999px'
              }}>
                <div style={{
                  width: '12px',
                  height: '12px',
                  backgroundColor: '#10b981',
                  borderRadius: '50%',
                  marginRight: '12px',
                  animation: 'pulse 2s infinite'
                }}></div>
                <span style={{ fontSize: '14px', fontWeight: '500' }}>Backend Connected</span>
              </div>
              <button
                onClick={getNewSample}
                disabled={isLoading}
                className="btn btn-primary"
                style={{ 
                  padding: '12px 24px',
                  opacity: isLoading ? '0.6' : '1',
                  cursor: isLoading ? 'not-allowed' : 'pointer'
                }}
              >
                {isLoading ? (
                  <div className="flex items-center">
                    <div style={{
                      width: '20px',
                      height: '20px',
                      border: '2px solid white',
                      borderTop: '2px solid transparent',
                      borderRadius: '50%',
                      marginRight: '8px',
                      animation: 'spin 1s linear infinite'
                    }}></div>
                    Loading...
                  </div>
                ) : (
                  'Get New Sample'
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container py-8">
        {error && (
          <div style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '12px',
            padding: '24px',
            marginBottom: '32px'
          }}>
            <div className="flex items-center">
              <div style={{ fontSize: '32px', marginRight: '16px' }}>⚠️</div>
              <div style={{ color: '#b91c1c', fontWeight: '500' }}>{error}</div>
            </div>
          </div>
        )}

        <div className="space-y-8">
          {/* Sample Viewer */}
          <SampleViewer sample={sample} sampleId={urlSampleId} />

          {/* User Decision */}
          <UserDecision
            userDecision={userDecision}
            onDecisionChange={(decision: UserDecisionType) => setUserDecision(decision)}
            onSubmit={handleSubmit}
            isSubmitting={isSubmitting}
            hasSample={!!sample}
          />

          {/* Risk Result */}
          <RiskResult
            prediction={prediction}
            userDecision={userDecision}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
