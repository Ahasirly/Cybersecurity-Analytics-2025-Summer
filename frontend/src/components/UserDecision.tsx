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

  const options = [
    { 
      value: 'Safe', 
      icon: '✅', 
      label: 'Safe', 
      description: 'I think this is normal behavior',
      color: '#10b981'
    },
    { 
      value: 'Unsafe', 
      icon: '❌', 
      label: 'Unsafe', 
      description: 'I think this is risky behavior',
      color: '#ef4444'
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

      <div style={{ display: 'flex', gap: '16px', marginBottom: '32px', justifyContent: 'center' }}>
        {options.map((option) => (
          <button
            key={option.value}
            onClick={() => onDecisionChange(option.value as UserDecisionType)}
            style={{
              border: userDecision === option.value ? `2px solid ${option.color}` : '2px solid #e5e7eb',
              backgroundColor: userDecision === option.value ? `${option.color}15` : 'white',
              borderRadius: '12px',
              padding: '16px 32px',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '18px',
              fontWeight: '600',
              minWidth: '150px',
              justifyContent: 'center'
            }}
            onMouseOver={(e) => {
              if (userDecision !== option.value) {
                e.currentTarget.style.backgroundColor = `${option.color}08`;
                e.currentTarget.style.borderColor = `${option.color}60`;
              }
            }}
            onMouseOut={(e) => {
              if (userDecision !== option.value) {
                e.currentTarget.style.backgroundColor = 'white';
                e.currentTarget.style.borderColor = '#e5e7eb';
              }
            }}
          >
            <span style={{ fontSize: '18px' }}>{option.icon}</span>
            <span style={{ color: userDecision === option.value ? option.color : '#374151' }}>
              {option.label}
            </span>
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