import React from 'react';
import { PredictionResponse, UserDecision, RiskLevel } from '../types';

interface RiskResultProps {
  prediction: PredictionResponse | null;
  userDecision: UserDecision | null;
}

const RiskResult: React.FC<RiskResultProps> = ({ prediction, userDecision }) => {
  if (!prediction) {
    return null;
  }

  const getRiskLevelColor = (level: RiskLevel) => {
    switch (level) {
      case 'Safe':
        return '#10b981';
      case 'Suspicious':
        return '#f59e0b';
      case 'High':
        return '#ef4444';
      case 'Critical':
        return '#dc2626';
      default:
        return '#6b7280';
    }
  };

  const getRiskLevelIcon = (level: RiskLevel) => {
    switch (level) {
      case 'Safe':
        return '✅';
      case 'Suspicious':
        return '⚠️';
      case 'High':
        return '🚨';
      case 'Critical':
        return '💀';
      default:
        return '❓';
    }
  };

  const getUserDecisionText = (decision: UserDecision) => {
    switch (decision) {
      case 'safe':
        return 'Safe';
      case 'suspicious':
        return 'Suspicious';
      case 'high':
        return 'High';
      case 'critical':
        return 'Critical';
      default:
        return '';
    }
  };

  const getUserDecisionColor = (decision: UserDecision) => {
    switch (decision) {
      case 'safe':
        return 'text-green-600';
      case 'suspicious':
        return 'text-yellow-600';
      case 'high':
        return 'text-red-600';
      case 'critical':
        return 'text-red-800';
      default:
        return 'text-gray-600';
    }
  };

  const isUserCorrect = () => {
    if (!userDecision) return null;
    
    // 直接比较用户选择和模型输出
    const userChoice = userDecision.charAt(0).toUpperCase() + userDecision.slice(1);
    return userChoice === prediction.final_risk_level;
  };

  const userCorrect = isUserCorrect();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Final Risk Assessment */}
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        borderRadius: '1rem',
        padding: '2rem',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        border: '1px solid rgba(229, 231, 235, 0.5)'
      }}>
        <h3 style={{
          fontSize: '1.875rem',
          fontWeight: '700',
          color: '#1f2937',
          marginBottom: '2rem',
          textAlign: 'center'
        }}>
          🎯 AI Risk Assessment Results
        </h3>
        
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '2rem' 
        }}>
          {/* User Assessment - Most Prominent */}
          {userDecision && (
            <div style={{
              background: userCorrect ? 'linear-gradient(135deg, #f0fdf4, #d1fae5)' : 'linear-gradient(135deg, #fef2f2, #fecaca)',
              borderRadius: '1.5rem',
              padding: '2rem',
              border: userCorrect ? '2px solid #10b981' : '2px solid #ef4444',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
              textAlign: 'center'
            }}>
              <div style={{ 
                fontSize: '3rem', 
                marginBottom: '1rem',
                color: userCorrect ? '#059669' : '#dc2626'
              }}>
              </div>
              <div style={{ 
                fontSize: '2rem', 
                fontWeight: '700',
                marginBottom: '1rem',
                color: userCorrect ? '#059669' : '#dc2626'
              }}>
                {userCorrect ? 'Correct Assessment!' : 'Incorrect Assessment'}
              </div>
              <div style={{ 
                fontSize: '1.25rem', 
                color: '#374151',
                marginBottom: '1.5rem'
              }}>
                Your Assessment: <span style={{ 
                  fontWeight: '700',
                  color: getUserDecisionColor(userDecision).includes('green') ? '#059669' :
                         getUserDecisionColor(userDecision).includes('yellow') ? '#d97706' :
                         getUserDecisionColor(userDecision).includes('red') ? '#dc2626' : '#374151'
                }}>{getUserDecisionText(userDecision)}</span>
              </div>
              <div style={{ 
                fontSize: '1.125rem', 
                color: '#6b7280'
              }}>
                AI's Assessment: <span style={{ 
                  fontWeight: '700',
                  color: getRiskLevelColor(prediction.final_risk_level)
                }}>{prediction.final_risk_level}</span>
              </div>
            </div>
          )}

          {/* AI Confidence */}
          <div style={{
            background: 'linear-gradient(to right, #eff6ff, #e0e7ff)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid rgba(191, 219, 254, 0.5)',
            width: '80%',
            margin: '0 auto'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <span style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1d4ed8' }}>SPAM Confidence</span>
              <span style={{ fontSize: '1.875rem', fontWeight: '700', color: '#1e40af' }}>
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ width: '100%', backgroundColor: '#bfdbfe', borderRadius: '9999px', height: '1rem' }}>
              <div 
                style={{ 
                  background: 'linear-gradient(to right, #3b82f6, #6366f1)',
                  height: '1rem',
                  borderRadius: '9999px',
                  transition: 'all 1s ease-out',
                  width: `${prediction.confidence * 100}%`
                }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Risk Scores */}
      <div style={{
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(10px)',
        borderRadius: '1rem',
        padding: '2rem',
        boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
        border: '1px solid rgba(229, 231, 235, 0.5)'
      }}>
        <h4 style={{
          fontSize: '1.5rem',
          fontWeight: '700',
          color: '#1f2937',
          marginBottom: '2rem',
          textAlign: 'center'
        }}>
          📊 Detailed Risk Analysis
        </h4>
        
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '1.5rem' 
        }}>
          <div style={{
            background: 'linear-gradient(to bottom right, #fef2f2, #fce7f3)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid rgba(254, 202, 202, 0.5)',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            transition: 'box-shadow 0.3s ease'
          }}>
            <div style={{ fontSize: '1.125rem', fontWeight: '600', color: '#b91c1c', marginBottom: '0.5rem' }}>URL Risk</div>
            <div style={{ fontSize: '1.875rem', fontWeight: '700', color: '#991b1b' }}>
              {(prediction.url_risk * 100).toFixed(1)}%
            </div>
          </div>
          
          <div style={{
            background: 'linear-gradient(to bottom right, #eff6ff, #e0e7ff)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid rgba(191, 219, 254, 0.5)',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            transition: 'box-shadow 0.3s ease'
          }}>
            <div style={{ fontSize: '1.125rem', fontWeight: '600', color: '#1d4ed8', marginBottom: '0.5rem' }}>Network Risk</div>
            <div style={{ fontSize: '1.875rem', fontWeight: '700', color: '#1e40af' }}>
              {(prediction.network_risk * 100).toFixed(1)}%
            </div>
          </div>
          
          <div style={{
            background: 'linear-gradient(to bottom right, #f0fdf4, #d1fae5)',
            borderRadius: '1rem',
            padding: '1.5rem',
            border: '1px solid rgba(187, 247, 208, 0.5)',
            boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
            transition: 'box-shadow 0.3s ease'
          }}>
            <div style={{ fontSize: '1.125rem', fontWeight: '600', color: '#059669', marginBottom: '0.5rem' }}>User Risk</div>
            <div style={{ fontSize: '1.875rem', fontWeight: '700', color: '#047857' }}>
              {(prediction.user_risk * 100).toFixed(1)}%
            </div>
          </div>
          

        </div>
      </div>


    </div>
  );
};

export default RiskResult; 