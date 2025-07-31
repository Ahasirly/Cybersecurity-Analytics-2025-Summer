// API Response Types
export interface SampleData {
  // URL features
  url_entropy: number;
  url_has_login: number;
  url_count_dot: number;
  url_isshorted: number;
  url_count_https: number;
  url_count_atrate: number;
  
  // User features
  login_attempts: number;
  session_duration: number;
  failed_logins: number;
  ip_reputation_score: number;
  
  // Network features
  flow_duration: number;
  pkt_len_std: number;
  flow_byts_per_sec: number;
  bwd_pkt_len_max: number;
  fwd_pkt_len_mean: number;
  fwd_pkts_per_sec: number;
  psh_flag_cnt: number;
  
  // Decoded categorical features (from one-hot encoding)
  protocol_type?: string;
  encryption_used?: string;
  browser_type?: string;
}

export interface RandomSampleResponse {
  sample: SampleData;
  url_sample_id: number;
  network_sample_id: number;
  user_sample_id: number;
}

export interface PredictionResponse {
  url_risk: number;
  network_risk: number;
  user_risk: number;
  final_risk_level: 'Safe' | 'Suspicious' | 'High' | 'Critical';
  confidence: number;
}

// User Decision Types
export type UserDecision = 'Safe' | 'Unsafe';

// Risk Level Types
export type RiskLevel = 'Safe' | 'Unsafe';

// Feature Categories
export interface FeatureCategory {
  title: string;
  features: Array<{
    name: string;
    value: number;
    description: string;
  }>;
} 