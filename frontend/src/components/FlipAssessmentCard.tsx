import React, { useState, useEffect } from 'react';
import { PredictionResponse, UserDecision as UserDecisionType } from '../types';

interface FlipAssessmentCardProps {
  userDecision: UserDecisionType | null;
  onDecisionChange: (decision: UserDecisionType) => void;
  onSubmit: () => void;
  isSubmitting: boolean;
  hasSample: boolean;
  prediction: PredictionResponse | null;
  onNewAssessment: () => void;
}

const FlipAssessmentCard: React.FC<FlipAssessmentCardProps> = ({
  userDecision,
  onDecisionChange,
  onSubmit,
  isSubmitting,
  hasSample,
  prediction,
  onNewAssessment
}) => {
  const [isFlipped, setIsFlipped] = useState(false);

  // Function to render markdown-like formatting
  const renderMarkdownText = (text: string): string => {
    let formattedText = text;
    
    // Convert **bold** to HTML
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #dc2626; font-weight: 700;">$1</strong>');
    
    // Convert `code` to HTML
    formattedText = formattedText.replace(/`(.*?)`/g, '<code style="background: #f3f4f6; color: #1f2937; padding: 2px 4px; border-radius: 3px; font-family: monospace; font-size: 0.9em;">$1</code>');
    
    // Convert > blockquotes to HTML
    formattedText = formattedText.replace(/^> (.+)$/gm, '<blockquote style="border-left: 4px solid #f59e0b; padding-left: 12px; margin: 8px 0; font-style: italic; color: #92400e;">$1</blockquote>');
    
    // Convert line breaks
    formattedText = formattedText.replace(/\n/g, '<br/>');
    
    return formattedText;
  };

  // Reset to front face when getting a new sample
  useEffect(() => {
    if (prediction === null) {
      setIsFlipped(false);
    }
  }, [prediction]);

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

  const handleSubmit = () => {
    onSubmit();
    setIsFlipped(true); // Flip to result face after submission
  };

  const handleNewAssessment = () => {
    onNewAssessment(); // Call parent component function to clear prediction and user decision
    setIsFlipped(false); // Flip back to front face
  };

  const isUserCorrect = () => {
    if (!userDecision || !prediction) return null;
    return userDecision === prediction.final_risk_level;
  };

  const userCorrect = isUserCorrect();

  return (
    <div style={{ perspective: '1000px', minHeight: '550px', marginBottom: '2rem' }}>
      <div
        style={{
          position: 'relative',
          width: '100%',
          minHeight: '550px',
          textAlign: 'center',
          transition: 'transform 0.6s',
          transformStyle: 'preserve-3d',
          transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)'
        }}
      >
        {/* Front Face - User Assessment */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            minHeight: '550px',
            backfaceVisibility: 'hidden',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '1rem',
            padding: '2rem',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            border: '1px solid rgba(229, 231, 235, 0.5)',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <div style={{
            fontSize: '1.5rem',
            fontWeight: '700',
            color: '#1f2937',
            marginBottom: '1rem'
          }}>
            What's your assessment?
          </div>
          
          <div style={{
            fontSize: '1rem',
            color: '#6b7280',
            marginBottom: '2rem'
          }}>
            Based on the features above, what do you think about this network behavior?
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

          <button
            onClick={handleSubmit}
            disabled={!userDecision || !hasSample || isSubmitting}
            style={{
              background: (!userDecision || !hasSample) ? '#d1d5db' : 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: 'white',
              border: 'none',
              borderRadius: '0.75rem',
              padding: '1rem 2rem',
              fontSize: '1.125rem',
              fontWeight: '600',
              cursor: (!userDecision || !hasSample) ? 'not-allowed' : 'pointer',
              transition: 'all 0.3s ease',
              opacity: isSubmitting ? 0.7 : 1
            }}
          >
            {isSubmitting ? 'Analyzing...' : 'Submit Assessment'}
          </button>
        </div>

        {/* Back Face - AI Prediction Results */}
        <div
          style={{
            position: 'absolute',
            width: '100%',
            minHeight: '550px',
            backfaceVisibility: 'hidden',
            transform: 'rotateY(180deg)',
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(10px)',
            borderRadius: '1rem',
            padding: '1.5rem',
            boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            border: '1px solid rgba(229, 231, 235, 0.5)',
            overflow: 'auto'
          }}
        >
          {prediction ? (
            <>
              {/* User Choice Result Feedback */}
              {userDecision && (
                <div style={{
                  textAlign: 'center',
                  marginBottom: '2rem',
                  padding: '2rem',
                  background: userCorrect === true 
                    ? 'linear-gradient(135deg, #f0fdf4, #dcfce7)' 
                    : 'linear-gradient(135deg, #f8fafc, #e2e8f0)',
                  borderRadius: '1rem',
                  border: userCorrect === true 
                    ? '2px solid #22c55e' 
                    : '2px solid #94a3b8',
                  boxShadow: userCorrect === true
                    ? '0 10px 15px -3px rgba(34, 197, 94, 0.2), 0 4px 6px -2px rgba(34, 197, 94, 0.1)'
                    : '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)'
                }}>
                  
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.5rem'
                  }}>
                    <div style={{
                      fontSize: '1.125rem',
                      fontWeight: '600',
                      color: '#4b5563'
                    }}>
                      Your Assessment
                    </div>
                    <div style={{
                      fontSize: '1.5rem',
                      color: userCorrect === true ? '#22c55e' : '#ef4444'
                    }}>
                      {userCorrect === true ? '✓' : '✗'}
                    </div>
                  </div>
                  
                  <div style={{
                    fontSize: '2.5rem',
                    fontWeight: '800',
                    color: userDecision === 'Safe' ? '#16a34a' : '#dc2626',
                    marginBottom: '0.5rem'
                  }}>
                    {userDecision}
                  </div>
                  
                  <div style={{
                    fontSize: '0.875rem',
                    color: '#6b7280',
                    fontStyle: 'italic'
                  }}>
                    {userDecision === 'Safe' 
                      ? 'You assessed this behavior as normal'
                      : 'You identified potential security concerns'
                    }
                  </div>
                </div>
              )}

                              {/* AI Assessment Result */}
              <div style={{
                textAlign: 'center',
                marginBottom: '2rem',
                padding: '2rem',
                background: (prediction.final_risk_level as string) === 'Unsafe' 
                  ? 'linear-gradient(135deg, #fef2f2, #fee2e2)'
                  : 'linear-gradient(135deg, #f0fdf4, #dcfce7)',
                borderRadius: '1rem',
                border: (prediction.final_risk_level as string) === 'Unsafe' 
                  ? '2px solid #ef4444'
                  : '2px solid #22c55e',
                boxShadow: (prediction.final_risk_level as string) === 'Unsafe'
                  ? '0 10px 15px -3px rgba(239, 68, 68, 0.2), 0 4px 6px -2px rgba(239, 68, 68, 0.1)'
                  : '0 10px 15px -3px rgba(34, 197, 94, 0.2), 0 4px 6px -2px rgba(34, 197, 94, 0.1)'
              }}>
                
                <div style={{
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  color: '#4b5563',
                  marginBottom: '0.5rem'
                }}>
                  AI Security Assessment
                </div>
                
                <div style={{
                  fontSize: '2.5rem',
                  fontWeight: '800',
                  color: prediction.final_risk_level === 'Safe' ? '#16a34a' : '#dc2626',
                  marginBottom: '0.5rem'
                }}>
                  {prediction.final_risk_level}
                </div>
                
                <div style={{
                  fontSize: '1.125rem',
                  fontWeight: '600',
                  color: '#4b5563',
                  marginBottom: '0.5rem'
                }}>
                  Confidence: {(prediction.confidence * 100).toFixed(1)}%
                </div>

                <div style={{
                  fontSize: '0.875rem',
                  color: '#6b7280',
                  fontStyle: 'italic'
                }}>
                  {prediction.final_risk_level === 'Safe' 
                    ? 'AI analysis indicates this behavior appears normal'
                    : 'AI analysis indicates potential security concerns'
                  }
                </div>
              </div>

              {/* Detailed Risk Analysis */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', 
                gap: '1rem',
                marginBottom: '2rem'
              }}>
                {[
                  { name: 'URL Risk', value: prediction.url_risk },
                  { name: 'User Risk', value: prediction.user_risk },
                  { name: 'Network Risk', value: prediction.network_risk }
                ].map((risk) => (
                  <div key={risk.name} style={{
                    background: 'white',
                    borderRadius: '12px',
                    padding: '1rem',
                    border: risk.value >= 0.5 ? '2px solid #ef4444' : '2px solid #22c55e',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                    textAlign: 'center'
                  }}>
                    <div style={{ 
                      fontSize: '0.875rem', 
                      fontWeight: '600', 
                      color: '#374151',
                      marginBottom: '0.5rem'
                    }}>
                      {risk.name}
                    </div>
                    <div style={{ 
                      fontSize: '1.5rem', 
                      fontWeight: '700', 
                      color: risk.value >= 0.5 ? '#ef4444' : '#22c55e'
                    }}>
                      {(risk.value * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* LLM Expert Analysis */}
              {prediction.llm_analysis && (
                <div style={{
                  background: 'linear-gradient(135deg, #fefce8, #fef3c7)',
                  borderRadius: '16px',
                  padding: '1.5rem',
                  border: '2px solid #f59e0b',
                  boxShadow: '0 8px 25px -5px rgba(245, 158, 11, 0.2), 0 4px 6px -2px rgba(245, 158, 11, 0.1)',
                  marginBottom: '1rem',
                  position: 'relative',
                  overflow: 'hidden'
                }}>
                  {/* Background decoration */}
                  <div style={{
                    position: 'absolute',
                    top: '-10px',
                    right: '-10px',
                    width: '60px',
                    height: '60px',
                    background: 'linear-gradient(45deg, #fbbf24, #f59e0b)',
                    borderRadius: '50%',
                    opacity: '0.1'
                  }}></div>
                  
                  <div style={{ 
                    fontSize: '1.1rem', 
                    fontWeight: '700', 
                    color: '#92400e',
                    marginBottom: '1rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    position: 'relative',
                    zIndex: 1
                  }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      background: 'linear-gradient(135deg, #fbbf24, #f59e0b)',
                      borderRadius: '50%',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '16px'
                    }}>
                      💡
                    </div>
                    Expert Security Insights
                  </div>
                  
                                     <div style={{ 
                     fontSize: '0.95rem', 
                     color: '#78350f',
                     lineHeight: '1.7',
                     position: 'relative',
                     zIndex: 1,
                     padding: '0.5rem',
                     background: 'rgba(255, 255, 255, 0.3)',
                     borderRadius: '8px',
                     border: '1px solid rgba(245, 158, 11, 0.2)'
                   }}
                   dangerouslySetInnerHTML={{
                     __html: renderMarkdownText(prediction.llm_analysis)
                   }}>
                   </div>
                </div>
              )}

            </>
          ) : (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              fontSize: '1.25rem',
              color: '#6b7280'
            }}>
              {isSubmitting ? (
                <>
                  {/* Loading Animation */}
                  <div style={{
                    width: '60px',
                    height: '60px',
                    border: '6px solid #e5e7eb',
                    borderTop: '6px solid #3b82f6',
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                    marginBottom: '1rem'
                  }}></div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#4b5563', marginBottom: '0.5rem' }}>
                    Analyzing Security Patterns...
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#6b7280', textAlign: 'center' }}>
                    Our AI is examining URL structure, network traffic,<br/>
                    and user behavior to provide expert insights
                  </div>
                </>
              ) : (
                <div>No prediction results yet</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default FlipAssessmentCard; 