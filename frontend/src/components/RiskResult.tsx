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
      case 'Unsafe':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getRiskLevelIcon = (level: RiskLevel) => {
    switch (level) {
      case 'Safe':
        return '✅';
      case 'Unsafe':
        return '⚠️';
      default:
        return '❓';
    }
  };

  const getUserDecisionText = (decision: UserDecision) => {
    switch (decision) {
      case 'Safe':
        return 'Safe';
      case 'Unsafe':
        return 'Unsafe';
      default:
        return '';
    }
  };

  const getUserDecisionColor = (decision: UserDecision) => {
    switch (decision) {
      case 'Safe':
        return 'text-green-600';
      case 'Unsafe':
        return 'text-red-600';
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

  const getRiskColorScheme = (riskValue: number) => {
    const riskPercentage = riskValue * 100;
    
    if (riskPercentage >= 80) {
      // Critical/High Risk - Red like medical abnormal values
      return {
        background: 'linear-gradient(to bottom right, #fef2f2, #fee2e2)',
        border: '2px solid #dc2626',
        textColor: '#dc2626',
        labelColor: '#b91c1c',
        boxShadow: '0 10px 15px -3px rgba(220, 38, 38, 0.2), 0 4px 6px -2px rgba(220, 38, 38, 0.1)',
        pulse: true
      };
    } else if (riskPercentage >= 60) {
      // High Risk - Orange/Red
      return {
        background: 'linear-gradient(to bottom right, #fef3c7, #fed7aa)',
        border: '1px solid #f59e0b',
        textColor: '#d97706',
        labelColor: '#b45309',
        boxShadow: '0 10px 15px -3px rgba(245, 158, 11, 0.15), 0 4px 6px -2px rgba(245, 158, 11, 0.1)',
        pulse: false
      };
    } else if (riskPercentage >= 30) {
      // Medium Risk - Yellow
      return {
        background: 'linear-gradient(to bottom right, #fefce8, #fef3c7)',
        border: '1px solid #eab308',
        textColor: '#ca8a04',
        labelColor: '#a16207',
        boxShadow: '0 10px 15px -3px rgba(234, 179, 8, 0.1), 0 4px 6px -2px rgba(234, 179, 8, 0.05)',
        pulse: false
      };
    } else {
      // Low Risk - Green (normal)
      return {
        background: 'linear-gradient(to bottom right, #f0fdf4, #d1fae5)',
        border: '1px solid rgba(187, 247, 208, 0.5)',
        textColor: '#047857',
        labelColor: '#059669',
        boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        pulse: false
      };
    }
  };

  const getRiskIcon = (riskValue: number) => {
    const riskPercentage = riskValue * 100;
    if (riskPercentage >= 80) return '🚨';
    if (riskPercentage >= 60) return '⚠️';
    if (riskPercentage >= 30) return '⚡';
    return '✅';
  };

  const userCorrect = isUserCorrect();

  return (
    <div style={{
      marginTop: '2rem',
      padding: '2rem',
      background: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(10px)',
      borderRadius: '1.5rem',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
      border: '1px solid rgba(255, 255, 255, 0.2)'
    }}>
      {/* Overall Risk Summary */}
      <div style={{
        textAlign: 'center',
        marginBottom: '2.5rem',
        padding: '2rem',
        background: prediction.final_risk_level === 'Safe' 
          ? 'linear-gradient(135deg, #f0fdf4, #dcfce7)' 
          : 'linear-gradient(135deg, #fef2f2, #fee2e2)',
        borderRadius: '1rem',
        border: prediction.final_risk_level === 'Safe' 
          ? '2px solid #22c55e' 
          : '2px solid #ef4444',
        position: 'relative',
        boxShadow: prediction.final_risk_level === 'Safe'
          ? '0 10px 15px -3px rgba(34, 197, 94, 0.2), 0 4px 6px -2px rgba(34, 197, 94, 0.1)'
          : '0 10px 15px -3px rgba(239, 68, 68, 0.2), 0 4px 6px -2px rgba(239, 68, 68, 0.1)'
      }}>
        <div style={{
          position: 'absolute',
          top: '1rem',
          right: '1rem',
          fontSize: '2rem'
        }}>
          {prediction.final_risk_level === 'Safe' ? '✅' : '❌'}
        </div>
        
        <div style={{
          fontSize: '1.125rem',
          fontWeight: '600',
          color: '#4b5563',
          marginBottom: '0.5rem'
        }}>
          Overall Security Assessment
        </div>
        
        <div style={{
          fontSize: '3rem',
          fontWeight: '800',
          color: prediction.final_risk_level === 'Safe' ? '#16a34a' : '#dc2626',
          marginBottom: '0.5rem'
        }}>
          {prediction.final_risk_level}
        </div>
        
        <div style={{
          fontSize: '1.25rem',
          fontWeight: '600',
          color: '#4b5563'
        }}>
          Confidence: {(prediction.confidence * 100).toFixed(1)}%
        </div>

        {userDecision && (
          <div style={{
            marginTop: '1.5rem',
            padding: '1rem',
            backgroundColor: userCorrect === true ? '#dcfce7' : userCorrect === false ? '#fef2f2' : '#f3f4f6',
            borderRadius: '0.75rem',
            border: `2px solid ${userCorrect === true ? '#16a34a' : userCorrect === false ? '#dc2626' : '#9ca3af'}`
          }}>
            <div style={{
              fontSize: '1rem',
              fontWeight: '600',
              color: userCorrect === true ? '#16a34a' : userCorrect === false ? '#dc2626' : '#6b7280'
            }}>
              Your Assessment: {getUserDecisionText(userDecision)}
              {userCorrect === true && ' ✅ Correct!'}
              {userCorrect === false && ' ❌ Different from AI'}
              {userCorrect === null && ' 📝 Recorded'}
            </div>
          </div>
        )}

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
          {/* URL Risk Card */}
          <div style={{
            background: 'white',
            borderRadius: '12px',
            padding: '1.5rem',
            border: prediction.url_risk >= 0.5 ? '2px solid #ef4444' : '2px solid #22c55e',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            transition: 'all 0.2s ease'
          }}>
            <div style={{ 
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '0.5rem'
            }}>
              <div style={{ 
                fontSize: '1.125rem', 
                fontWeight: '600', 
                color: '#374151'
              }}>
                URL Risk
              </div>
              <div style={{ fontSize: '1.5rem' }}>
                {prediction.url_risk >= 0.5 ? '🔴' : '🟢'}
              </div>
            </div>
            <div style={{ 
              fontSize: '2rem', 
              fontWeight: '700', 
              color: prediction.url_risk >= 0.5 ? '#ef4444' : '#22c55e',
              marginBottom: '0.5rem'
            }}>
              {(prediction.url_risk * 100).toFixed(1)}%
            </div>
            <div style={{ 
              fontSize: '0.875rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              {prediction.url_risk >= 0.5 ? 'High risk detected' : 'Low risk detected'}
            </div>
          </div>
          
          {/* Network Risk Card */}
          <div style={{
            background: 'white',
            borderRadius: '12px',
            padding: '1.5rem',
            border: prediction.network_risk >= 0.5 ? '2px solid #ef4444' : '2px solid #22c55e',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            transition: 'all 0.2s ease'
          }}>
            <div style={{ 
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '0.5rem'
            }}>
              <div style={{ 
                fontSize: '1.125rem', 
                fontWeight: '600', 
                color: '#374151'
              }}>
                Network Risk
              </div>
              <div style={{ fontSize: '1.5rem' }}>
                {prediction.network_risk >= 0.5 ? '🔴' : '🟢'}
              </div>
            </div>
            <div style={{ 
              fontSize: '2rem', 
              fontWeight: '700', 
              color: prediction.network_risk >= 0.5 ? '#ef4444' : '#22c55e',
              marginBottom: '0.5rem'
            }}>
              {(prediction.network_risk * 100).toFixed(1)}%
            </div>
            <div style={{ 
              fontSize: '0.875rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              {prediction.network_risk >= 0.5 ? 'High risk detected' : 'Low risk detected'}
            </div>
          </div>
          
          {/* User Risk Card */}
          <div style={{
            background: 'white',
            borderRadius: '12px',
            padding: '1.5rem',
            border: prediction.user_risk >= 0.5 ? '2px solid #ef4444' : '2px solid #22c55e',
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            transition: 'all 0.2s ease'
          }}>
            <div style={{ 
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '0.5rem'
            }}>
              <div style={{ 
                fontSize: '1.125rem', 
                fontWeight: '600', 
                color: '#374151'
              }}>
                User Risk
              </div>
              <div style={{ fontSize: '1.5rem' }}>
                {prediction.user_risk >= 0.5 ? '🔴' : '🟢'}
              </div>
            </div>
            <div style={{ 
              fontSize: '2rem', 
              fontWeight: '700', 
              color: prediction.user_risk >= 0.5 ? '#ef4444' : '#22c55e',
              marginBottom: '0.5rem'
            }}>
              {(prediction.user_risk * 100).toFixed(1)}%
            </div>
            <div style={{ 
              fontSize: '0.875rem', 
              color: '#6b7280',
              fontWeight: '500'
            }}>
              {prediction.user_risk >= 0.5 ? 'High risk detected' : 'Low risk detected'}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
};

export default RiskResult; 