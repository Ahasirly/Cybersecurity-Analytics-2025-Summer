import React from 'react';
import { UserDecision as UserDecisionType } from '../types';

interface UserDecisionProps {
  userDecision: UserDecisionType | null;
  onDecisionChange: (decision: UserDecisionType) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  hasSample: boolean;
}

const UserDecision: React.FC<UserDecisionProps> = ({
  userDecision,
  onDecisionChange,
  onSubmit,
  isSubmitting,
  hasSample
}) => {
  if (!hasSample) {
    return null;
  }

  const decisions = [
    {
      value: 'safe' as UserDecisionType,
      label: 'Safe',
      description: 'I think this is normal network behavior',
      icon: '✅',
      color: '#10b981'
    },
    {
      value: 'suspicious' as UserDecisionType,
      label: 'Suspicious',
      description: 'I think this might be suspicious behavior',
      icon: '⚠️',
      color: '#f59e0b'
    },
    {
      value: 'high' as UserDecisionType,
      label: 'High',
      description: 'I think this is high risk behavior',
      icon: '🚨',
      color: '#ef4444'
    },
    {
      value: 'critical' as UserDecisionType,
      label: 'Critical',
      description: 'I think this is critical risk behavior',
      icon: '💀',
      color: '#dc2626'
    }
  ];

  return (
    <div className="card p-8">
      <div className="text-center mb-8">
        <div style={{ fontSize: '48px', marginBottom: '16px' }}></div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">What's your assessment?</h2>
        <p className="text-gray-600" style={{ fontSize: '16px' }}>
          Based on the features above, what do you think about this network behavior?
        </p>
      </div>

      <div className="grid grid-cols-4" style={{ gap: '16px', marginBottom: '32px' }}>
        {decisions.map((decision) => (
          <button
            key={decision.value}
            onClick={() => onDecisionChange(decision.value)}
            className="card p-6 text-center transition-all duration-300"
            style={{
              border: userDecision === decision.value ? `3px solid ${decision.color}` : '1px solid rgba(229, 231, 235, 0.5)',
              backgroundColor: userDecision === decision.value ? `${decision.color}10` : 'rgba(255, 255, 255, 0.8)',
              transform: userDecision === decision.value ? 'scale(1.02)' : 'scale(1)',
              cursor: 'pointer',
              minHeight: '200px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>{decision.icon}</div>
            <h3 className="text-xl font-bold mb-3" style={{ color: decision.color }}>
              {decision.label}
            </h3>
            <p className="text-gray-600" style={{ fontSize: '14px', lineHeight: '1.5' }}>
              {decision.description}
            </p>
          </button>
        ))}
      </div>

      <div className="text-center">
        <button
          onClick={onSubmit}
          disabled={!userDecision || isSubmitting}
          className="btn btn-primary"
          style={{
            padding: '16px 48px',
            fontSize: '18px',
            fontWeight: '600',
            opacity: !userDecision || isSubmitting ? '0.5' : '1',
            cursor: !userDecision || isSubmitting ? 'not-allowed' : 'pointer',
            background: userDecision ? 'linear-gradient(135deg, #3b82f6, #8b5cf6)' : 'linear-gradient(135deg, #9ca3af, #6b7280)'
          }}
        >
          {isSubmitting ? (
            <div className="flex items-center">
              <div style={{
                width: '24px',
                height: '24px',
                border: '3px solid white',
                borderTop: '3px solid transparent',
                borderRadius: '50%',
                marginRight: '12px',
                animation: 'spin 1s linear infinite'
              }}></div>
              Analyzing...
            </div>
          ) : (
            'Submit Assessment'
          )}
        </button>
      </div>
    </div>
  );
};

export default UserDecision; 