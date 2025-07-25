import React, { useState } from 'react';
import { SampleData } from '../types';

interface SampleViewerProps {
  sample: SampleData | null;
  sampleId: number | null;
}

interface FeatureDialogProps {
  isOpen: boolean;
  onClose: () => void;
  feature: {
    name: string;
    displayName: string;
    description: string;
    value: number | string;
    guidance: string;
  } | null;
}

interface ConfigurationDialogProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  features: Array<{
    name: string;
    displayName: string;
    description: string;
    value: number | string;
  }>;
}

const FeatureDialog: React.FC<FeatureDialogProps> = ({ isOpen, onClose, feature }) => {
  if (!isOpen || !feature) return null;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-content" onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h2 className="dialog-title">{feature.displayName}</h2>
          <button className="dialog-close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="dialog-body">
          <div className="dialog-value">
            {typeof feature.value === 'string' ? feature.value : feature.value.toFixed(4)}
          </div>
          <div className="dialog-description">{feature.description}</div>
          <div className="dialog-guidance">
            <div className="guidance-title">Security Guidance</div>
            <div className="guidance-text">{feature.guidance}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ConfigurationDialog: React.FC<ConfigurationDialogProps> = ({ isOpen, onClose, title, features }) => {
  if (!isOpen) return null;

  return (
    <div className="dialog-overlay" onClick={onClose}>
      <div className="dialog-content" style={{ maxWidth: '800px', maxHeight: '80vh', overflow: 'auto' }} onClick={(e) => e.stopPropagation()}>
        <div className="dialog-header">
          <h2 className="dialog-title">{title} - Full Configuration</h2>
          <button className="dialog-close" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="dialog-body">
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
            gap: '1rem',
            maxHeight: '60vh',
            overflow: 'auto'
          }}>
            {features.map((feature, index) => {
              // 检查是否是教学特征（前几个特征）
              const isTeachingFeature = index < (title === 'URL Features' ? 6 : title === 'User Features' ? 7 : 7);
              
              return (
                <div key={feature.name} style={{
                  background: isTeachingFeature ? '#fef3c7' : '#f8fafc',
                  border: '1px solid #e2e8f0',
                  borderRadius: '0.75rem',
                  padding: '1rem',
                  borderLeft: `4px solid ${isTeachingFeature ? '#f59e0b' : '#3b82f6'}`
                }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '0.5rem' 
                  }}>
                    <div style={{ fontWeight: '600', color: '#1f2937' }}>
                      {feature.displayName}
                    </div>
                    {isTeachingFeature && (
                      <div style={{
                        background: '#f59e0b',
                        color: 'white',
                        fontSize: '0.75rem',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '0.375rem',
                        fontWeight: '500'
                      }}>
                        Crucial
                      </div>
                    )}
                  </div>
                  <div style={{ color: '#6b7280', fontSize: '0.875rem', marginBottom: '0.5rem' }}>
                    {feature.description}
                  </div>
                  <div style={{ 
                    fontFamily: 'Courier New, monospace',
                    fontSize: '1.25rem',
                    fontWeight: '700',
                    color: isTeachingFeature ? '#f59e0b' : '#3b82f6'
                  }}>
                    {typeof feature.value === 'string' ? feature.value : feature.value.toFixed(4)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

const SampleViewer: React.FC<SampleViewerProps> = ({ sample, sampleId }) => {
  const [selectedFeature, setSelectedFeature] = useState<{
    name: string;
    displayName: string;
    description: string;
    value: number | string;
    guidance: string;
  } | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [configDialogData, setConfigDialogData] = useState<{
    title: string;
    features: Array<{
      name: string;
      displayName: string;
      description: string;
      value: number | string;
    }>;
  } | null>(null);

  if (!sample) {
    return (
      <div className="card p-8 text-center" style={{ minHeight: '400px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        <div style={{ fontSize: '64px', marginBottom: '24px', opacity: '0.6' }}>🔍</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Ready to Start Learning?</h2>
        <p className="text-gray-600" style={{ fontSize: '18px', maxWidth: '500px' }}>
          Click the "Get New Sample" button above to begin your cybersecurity training session
        </p>
      </div>
    );
  }

  const urlFeatures = [
    { name: 'url_entropy', description: 'URL randomness/complexity score' },
    { name: 'url_count_dot', description: 'Number of dots in URL' },
    { name: 'url_len', description: 'Total length of URL' },
    { name: 'url_count_hyphen', description: 'Number of hyphens in URL' },
    { name: 'url_count_letter', description: 'Number of letters in URL' },
    { name: 'url_count_digit', description: 'Number of digits in URL' }
  ];

  const userFeatures = [
    { name: 'protocol_type', description: 'Network protocol type' },
    { name: 'encryption_used', description: 'Encryption method used' },
    { name: 'browser_type', description: 'Web browser type' },
    { name: 'login_attempts', description: 'Total number of login attempts' },
    { name: 'session_duration', description: 'User session duration (seconds)' },
    { name: 'ip_reputation_score', description: 'IP reputation score (0-1)' },
    { name: 'failed_logins', description: 'Number of failed login attempts' }
  ];

  const networkFeatures = [
    { name: 'Flow Duration', description: 'Network flow duration (microseconds)' },
    { name: 'Tot Fwd Pkts', description: 'Total packets sent to destination' },
    { name: 'Flow Pkts/s', description: 'Packet transfer rate (packets/second)' },
    { name: 'Fwd Pkt Len Max', description: 'Maximum size of outgoing packets' },
    { name: 'Pkt Len Mean', description: 'Average packet size across all packets' },
    { name: 'Pkt Size Avg', description: 'Average packet size in bytes' },
    { name: 'Flow Byts/s', description: 'Data transfer rate (bytes/second)' }
  ];

  // 完整的特征配置数据（包含教学特征）
  const fullUrlFeatures = [
    // 教学特征 - 始终显示
    { name: 'url_entropy', description: 'URL randomness/complexity score' },
    { name: 'url_count_dot', description: 'Number of dots in URL' },
    { name: 'url_len', description: 'Total length of URL' },
    { name: 'url_count_hyphen', description: 'Number of hyphens in URL' },
    { name: 'url_count_letter', description: 'Number of letters in URL' },
    { name: 'url_count_digit', description: 'Number of digits in URL' },
    // 其他完整特征
    { name: 'url_has_login', description: 'Whether URL contains login-related keywords' },
    { name: 'url_has_client', description: 'Whether URL contains client-related keywords' },
    { name: 'url_has_server', description: 'Whether URL contains server-related keywords' },
    { name: 'url_has_admin', description: 'Whether URL contains admin-related keywords' },
    { name: 'url_has_ip', description: 'Whether URL contains IP address' },
    { name: 'url_isshorted', description: 'Whether it is a shortened URL service' },
    { name: 'url_hamming_1', description: 'URL Hamming distance 1' },
    { name: 'url_hamming_00', description: 'URL Hamming distance 00' },
    { name: 'url_hamming_10', description: 'URL Hamming distance 10' },
    { name: 'url_hamming_01', description: 'URL Hamming distance 01' },
    { name: 'url_hamming_11', description: 'URL Hamming distance 11' },
    { name: 'url_2bentropy', description: 'URL 2-bit entropy' },
    { name: 'url_3bentropy', description: 'URL 3-bit entropy' },
    { name: 'url_count_dot', description: 'Number of dots in URL' },
    { name: 'url_count_https', description: 'Number of HTTPS protocol mentions' },
    { name: 'url_count_http', description: 'Number of HTTP protocol mentions' },
    { name: 'url_count_perc', description: 'Number of percent signs in URL' },
    { name: 'url_count_hyphen', description: 'Number of hyphens in URL' },
    { name: 'url_count_www', description: 'Number of www occurrences' },
    { name: 'url_count_atrate', description: 'Number of @ symbols in URL' },
    { name: 'url_count_hash', description: 'Number of hash symbols in URL' },
    { name: 'url_count_semicolon', description: 'Number of semicolons in URL' },
    { name: 'url_count_underscore', description: 'Number of underscores in URL' },
    { name: 'url_count_ques', description: 'Number of question marks in URL' },
    { name: 'url_count_equal', description: 'Number of equal signs in URL' },
    { name: 'url_count_amp', description: 'Number of ampersands in URL' },
    { name: 'url_count_letter', description: 'Number of letters in URL' },
    { name: 'url_count_digit', description: 'Number of digits in URL' },
    { name: 'url_count_sensitive_financial_words', description: 'Number of sensitive financial words' },
    { name: 'url_count_sensitive_words', description: 'Number of sensitive words' },
    { name: 'url_nunique_chars_ratio', description: 'Ratio of unique characters in URL' },
    { name: 'path_len', description: 'Length of URL path' },
    { name: 'path_count_no_of_dir', description: 'Number of directories in path' },
    { name: 'path_count_no_of_embed', description: 'Number of embedded objects in path' },
    { name: 'path_count_zero', description: 'Number of zero values in path' },
    { name: 'path_count_pertwent', description: 'Number of percent-twenty in path' },
    { name: 'path_has_any_sensitive_words', description: 'Whether path contains sensitive words' },
    { name: 'path_count_lower', description: 'Number of lowercase letters in path' },
    { name: 'path_count_upper', description: 'Number of uppercase letters in path' },
    { name: 'path_count_nonascii', description: 'Number of non-ASCII characters in path' },
    { name: 'path_has_singlechardir', description: 'Whether path has single character directories' },
    { name: 'path_has_upperdir', description: 'Whether path has upper directories' },
    { name: 'query_len', description: 'Length of URL query' },
    { name: 'query_count_components', description: 'Number of query components' },
    { name: 'pdomain_len', description: 'Length of primary domain' },
    { name: 'pdomain_count_hyphen', description: 'Number of hyphens in primary domain' },
    { name: 'pdomain_count_atrate', description: 'Number of @ symbols in primary domain' },
    { name: 'pdomain_count_non_alphanum', description: 'Number of non-alphanumeric characters in primary domain' },
    { name: 'pdomain_count_digit', description: 'Number of digits in primary domain' },
    { name: 'tld_len', description: 'Length of top-level domain' },
    { name: 'tld_is_sus', description: 'Whether top-level domain is suspicious' },
    { name: 'pdomain_min_distance', description: 'Minimum distance of primary domain' },
    { name: 'subdomain_len', description: 'Length of subdomain' },
    { name: 'subdomain_count_dot', description: 'Number of dots in subdomain' }
  ];

  const fullUserFeatures = [
    // 教学特征 - 始终显示
    { name: 'login_attempts', description: 'Total number of login attempts' },
    { name: 'session_duration', description: 'User session duration (seconds)' },
    { name: 'failed_logins', description: 'Number of failed login attempts' },
    { name: 'login_failure_rate', description: 'Percentage of failed logins' },
    { name: 'ip_reputation_score', description: 'IP reputation score (0-1)' },
    // 其他完整特征
    { name: 'network_packet_size', description: 'Size of network packets' },
    { name: 'unusual_time_access', description: 'Whether accessed at unusual times' }
  ];

  const fullNetworkFeatures = [
    // 教学特征 - 始终显示
    { name: 'Flow Duration', description: 'Network flow duration (microseconds)' },
    { name: 'Tot Fwd Pkts', description: 'Total packets sent to destination' },
    { name: 'Flow Pkts/s', description: 'Packet transfer rate (packets/second)' },
    { name: 'Fwd Pkt Len Max', description: 'Maximum size of outgoing packets' },
    { name: 'Pkt Len Mean', description: 'Average packet size across all packets' },
    { name: 'Pkt Size Avg', description: 'Average packet size in bytes' },
    { name: 'Flow Byts/s', description: 'Data transfer rate (bytes/second)' },
    // 其他完整特征
    { name: 'ACK Flag Cnt', description: 'Number of ACK flags' },
    { name: 'Active Max', description: 'Maximum active time' },
    { name: 'Active Mean', description: 'Mean active time' },
    { name: 'Active Min', description: 'Minimum active time' },
    { name: 'Active Std', description: 'Standard deviation of active time' },
    { name: 'Bwd Blk Rate Avg', description: 'Average backward block rate' },
    { name: 'Bwd Byts/b Avg', description: 'Average backward bytes per burst' },
    { name: 'Bwd Header Len', description: 'Backward header length' },
    { name: 'Bwd IAT Max', description: 'Maximum backward inter-arrival time' },
    { name: 'Bwd IAT Mean', description: 'Mean backward inter-arrival time' },
    { name: 'Bwd IAT Min', description: 'Minimum backward inter-arrival time' },
    { name: 'Bwd IAT Std', description: 'Standard deviation of backward inter-arrival time' },
    { name: 'Bwd IAT Tot', description: 'Total backward inter-arrival time' },
    { name: 'Bwd PSH Flags', description: 'Backward PSH flags' },
    { name: 'Bwd Pkt Len Max', description: 'Maximum backward packet length' },
    { name: 'Bwd Pkt Len Mean', description: 'Mean backward packet length' },
    { name: 'Bwd Pkt Len Min', description: 'Minimum backward packet length' },
    { name: 'Bwd Pkt Len Std', description: 'Standard deviation of backward packet length' },
    { name: 'Bwd Pkts/b Avg', description: 'Average backward packets per burst' },
    { name: 'Bwd Pkts/s', description: 'Backward packets per second' },
    { name: 'Bwd Seg Size Avg', description: 'Average backward segment size' },
    { name: 'Bwd URG Flags', description: 'Backward URG flags' },
    { name: 'CWE Flag Count', description: 'CWE flag count' },
    { name: 'Down/Up Ratio', description: 'Down/Up ratio' },
    { name: 'Dst Port', description: 'Destination port' },
    { name: 'ECE Flag Cnt', description: 'Number of ECE flags' },
    { name: 'FIN Flag Cnt', description: 'Number of FIN flags' },
    { name: 'Flow Byts/s', description: 'Flow bytes per second' },
    { name: 'Flow Duration', description: 'Flow duration' },
    { name: 'Flow IAT Max', description: 'Maximum flow inter-arrival time' },
    { name: 'Flow IAT Mean', description: 'Mean flow inter-arrival time' },
    { name: 'Flow IAT Min', description: 'Minimum flow inter-arrival time' },
    { name: 'Flow IAT Std', description: 'Standard deviation of flow inter-arrival time' },
    { name: 'Flow Pkts/s', description: 'Flow packets per second' },
    { name: 'Fwd Act Data Pkts', description: 'Forward active data packets' },
    { name: 'Fwd Blk Rate Avg', description: 'Average forward block rate' },
    { name: 'Fwd Byts/b Avg', description: 'Average forward bytes per burst' },
    { name: 'Fwd Header Len', description: 'Forward header length' },
    { name: 'Fwd IAT Max', description: 'Maximum forward inter-arrival time' },
    { name: 'Fwd IAT Mean', description: 'Mean forward inter-arrival time' },
    { name: 'Fwd IAT Min', description: 'Minimum forward inter-arrival time' },
    { name: 'Fwd IAT Std', description: 'Standard deviation of forward inter-arrival time' },
    { name: 'Fwd IAT Tot', description: 'Total forward inter-arrival time' },
    { name: 'Fwd PSH Flags', description: 'Forward PSH flags' },
    { name: 'Fwd Pkt Len Max', description: 'Maximum forward packet length' },
    { name: 'Fwd Pkt Len Mean', description: 'Mean forward packet length' },
    { name: 'Fwd Pkt Len Min', description: 'Minimum forward packet length' },
    { name: 'Fwd Pkt Len Std', description: 'Standard deviation of forward packet length' },
    { name: 'Fwd Pkts/b Avg', description: 'Average forward packets per burst' },
    { name: 'Fwd Pkts/s', description: 'Forward packets per second' },
    { name: 'Fwd Seg Size Avg', description: 'Average forward segment size' },
    { name: 'Fwd Seg Size Min', description: 'Minimum forward segment size' },
    { name: 'Fwd URG Flags', description: 'Forward URG flags' },
    { name: 'Idle Max', description: 'Maximum idle time' },
    { name: 'Idle Mean', description: 'Mean idle time' },
    { name: 'Idle Min', description: 'Minimum idle time' },
    { name: 'Idle Std', description: 'Standard deviation of idle time' },
    { name: 'Init Bwd Win Byts', description: 'Initial backward window bytes' },
    { name: 'Init Fwd Win Byts', description: 'Initial forward window bytes' },
    { name: 'PSH Flag Cnt', description: 'Number of PSH flags' },
    { name: 'Pkt Len Max', description: 'Maximum packet length' },
    { name: 'Pkt Len Mean', description: 'Mean packet length' },
    { name: 'Pkt Len Min', description: 'Minimum packet length' },
    { name: 'Pkt Len Std', description: 'Standard deviation of packet length' },
    { name: 'Pkt Len Var', description: 'Variance of packet length' },
    { name: 'Pkt Size Avg', description: 'Average packet size' },
    { name: 'Protocol', description: 'Protocol type' },
    { name: 'RST Flag Cnt', description: 'Number of RST flags' },
    { name: 'SYN Flag Cnt', description: 'Number of SYN flags' },
    { name: 'Subflow Bwd Byts', description: 'Subflow backward bytes' },
    { name: 'Subflow Bwd Pkts', description: 'Subflow backward packets' },
    { name: 'Subflow Fwd Byts', description: 'Subflow forward bytes' },
    { name: 'Subflow Fwd Pkts', description: 'Subflow forward packets' },
    { name: 'Tot Bwd Pkts', description: 'Total backward packets' },
    { name: 'Tot Fwd Pkts', description: 'Total forward packets' },
    { name: 'TotLen Bwd Pkts', description: 'Total length of backward packets' },
    { name: 'TotLen Fwd Pkts', description: 'Total length of forward packets' },
    { name: 'URG Flag Cnt', description: 'Number of URG flags' }
  ];

  const handleFeatureClick = (feature: { name: string; description: string }, displayName: string) => {
    const value = sample[feature.name as keyof SampleData] || 0;
    const guidance = getSecurityGuidance(feature.name);
    
    setSelectedFeature({
      name: feature.name,
      displayName,
      description: feature.description,
      value: value, // Keep original value type (string or number)
      guidance
    });
    setIsDialogOpen(true);
  };

  const handleShowConfiguration = (type: 'url' | 'user' | 'network') => {
    let title = '';
    let features: Array<{ name: string; displayName: string; description: string; value: number | string }> = [];

    switch (type) {
      case 'url':
        title = 'URL Features';
        features = fullUrlFeatures.map(f => ({
          ...f,
          displayName: f.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          value: (() => {
            const val = sample[f.name as keyof SampleData];
            return val || 0;
          })()
        }));
        break;
      case 'user':
        title = 'User Features';
        features = fullUserFeatures.map(f => ({
          ...f,
          displayName: f.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          value: (() => {
            const val = sample[f.name as keyof SampleData];
            return val || 0;
          })()
        }));
        break;
      case 'network':
        title = 'Network Features';
        features = fullNetworkFeatures.map(f => ({
          ...f,
          displayName: f.name,
          value: (() => {
            const val = sample[f.name as keyof SampleData];
            return val || 0;
          })()
        }));
        break;
    }

    setConfigDialogData({ title, features });
    setIsConfigDialogOpen(true);
  };

  const getDisplayName = (name: string) => {
    const nameMap: { [key: string]: string } = {
      'url_entropy': 'URL Complexity',
      'url_count_dot': 'Dot Count',
      'url_len': 'URL Length',
      'url_count_hyphen': 'Hyphen Count',
      'url_count_letter': 'Letter Count',
      'url_count_digit': 'Digit Count',
      'login_attempts': 'Login Attempts',
      'session_duration': 'Session Duration',
      'failed_logins': 'Failed Logins',

      'protocol_type': 'Protocol Type',
      'encryption_used': 'Encryption Used',
      'browser_type': 'Browser Type',
      'ip_reputation_score': 'IP Reputation Score',
      'Flow Duration': 'Flow Duration',
      'Tot Fwd Pkts': 'Outgoing Packets',
      'Flow Pkts/s': 'Packet Transfer Rate',
      'Fwd Pkt Len Max': 'Max Outgoing Size',
      'Pkt Len Mean': 'Average Packet Size',
      'Pkt Size Avg': 'Packet Size Average',
      'Flow Byts/s': 'Data Transfer Rate'
    };
    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getSecurityGuidance = (name: string) => {
    const guidanceMap: { [key: string]: string } = {
      'url_entropy': 'URL complexity score (0-10). Higher values indicate more complex/random URLs. Values above 4.0 often indicate obfuscated or malicious URLs.',
      'url_count_dot': 'Number of dots in URL. More dots can indicate subdomain manipulation. Legitimate URLs typically have 1-2 dots, while malicious ones may have 3+ dots.',
      'url_len': 'Total length of URL in characters. Very long URLs (>100 chars) may contain malicious parameters or obfuscation.',
      'url_count_hyphen': 'Number of hyphens in URL. Excessive hyphens can indicate domain spoofing attempts. Legitimate domains rarely have more than 2-3 hyphens.',
      'url_count_letter': 'Number of letters in URL. Normal URLs should have a reasonable letter count. Extremely high or low letter counts may indicate suspicious URLs.',
      'url_count_digit': 'Number of digits in URL. Many digits may indicate suspicious or generated URLs. Legitimate URLs typically have fewer digits than letters.',
      'login_attempts': 'Total number of login attempts. High number of attempts may indicate brute force attacks. More than 5 attempts in a short period is suspicious.',
      'session_duration': 'User session duration in seconds. Very short sessions may indicate automated attacks. Sessions under 30 seconds are often suspicious.',
      'failed_logins': 'Number of failed login attempts. Multiple failed logins suggest potential security threats. 3+ failed attempts indicate possible attack.',

      'protocol_type': 'Network protocol used (TCP/UDP/ICMP). Different protocols have different security implications.',
      'encryption_used': 'Encryption method (AES/DES/Unknown). Stronger encryption (AES) is generally more secure.',
      'browser_type': 'Web browser used (Chrome/Firefox/Safari/Edge). Some browsers may have different security vulnerabilities.',
      'ip_reputation_score': 'IP reputation score (0-1). Lower scores indicate IPs with poor reputation. Scores below 0.3 suggest malicious IP addresses.',
      'Flow Duration': 'Network flow duration in microseconds. Very short flows may indicate scanning or attack attempts. Flows under 1,000,000 microseconds are often suspicious.',
      'Tot Fwd Pkts': 'Total forward packets. Unusual packet counts may indicate abnormal traffic patterns. Very high or very low counts are suspicious.',
      'Flow Pkts/s': 'Flow packets per second. High packet rates may indicate DDoS or scanning activity. Rates above 1000 packets/second are concerning.',
      'Fwd Pkt Len Max': 'Maximum forward packet length in bytes. Very large packets may indicate data exfiltration. Packets over 1500 bytes should be investigated.',
      'Pkt Len Mean': 'Mean packet length in bytes. Abnormal packet sizes may indicate malicious payloads. Very small or very large average sizes are suspicious.',
      'Pkt Size Avg': 'Average packet size in bytes. Consistent small packets may indicate scanning behavior. Average sizes under 50 bytes are often suspicious.',
      'Flow Byts/s': 'Flow bytes per second. High data transfer rates may indicate data theft or DDoS. Rates above 1MB/second should be investigated.'
    };
    return guidanceMap[name] || 'This feature value is now displayed in its original scale for better understanding.';
  };

  const renderFeatureCard = (title: string, features: Array<{ name: string; description: string }>, color: string, type: 'url' | 'user' | 'network') => (
    <div className="card p-6" style={{ height: 'fit-content' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <h3 className="text-xl font-bold" style={{ color }}>
          {title}
        </h3>
        <button
          onClick={() => handleShowConfiguration(type)}
          style={{
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            color: 'white',
            border: 'none',
            borderRadius: '0.5rem',
            padding: '0.5rem 1rem',
            fontSize: '0.875rem',
            fontWeight: '500',
            cursor: 'pointer',
            transition: 'all 0.2s ease'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-1px)';
            e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = 'none';
          }}
        >
          Show Full Configuration
        </button>
      </div>
      <div className="grid grid-cols-2" style={{ gap: '1rem' }}>
        {features.map((feature) => {
          const displayName = getDisplayName(feature.name);
          const value = sample[feature.name as keyof SampleData] || 0;

          return (
            <div 
              key={feature.name} 
              className="feature-card"
              onClick={() => handleFeatureClick(feature, displayName)}
            >
              <div className="font-semibold text-gray-800 mb-1" style={{ fontSize: '14px' }}>
                {displayName}
              </div>
              <div className="text-gray-600 mb-2" style={{ fontSize: '12px' }}>
                {feature.description}
              </div>
              <div className="font-mono text-lg font-bold" style={{ color }}>
                {typeof value === 'string' ? value : (typeof value === 'number' ? value.toFixed(4) : (parseFloat(String(value)) || 0).toFixed(4))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="card p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold text-gray-800">
            Sample #{sampleId}
          </h2>
          <div style={{
            backgroundColor: '#f3f4f6',
            padding: '8px 16px',
            borderRadius: '9999px',
            fontSize: '14px',
            fontWeight: '500',
            color: '#6b7280'
          }}>
            Analysis Ready
          </div>
        </div>
        <p className="text-gray-600" style={{ fontSize: '16px' }}>
          Review the features below and make your assessment about the security risk level
        </p>
      </div>

      <div className="grid grid-cols-3" style={{ gap: '24px' }}>
        {renderFeatureCard('URL Features', urlFeatures, '#3b82f6', 'url')}
        {renderFeatureCard('User Features', userFeatures, '#10b981', 'user')}
        {renderFeatureCard('Network Features', networkFeatures, '#8b5cf6', 'network')}
      </div>

      <FeatureDialog 
        isOpen={isDialogOpen}
        onClose={() => setIsDialogOpen(false)}
        feature={selectedFeature}
      />

      <ConfigurationDialog
        isOpen={isConfigDialogOpen}
        onClose={() => setIsConfigDialogOpen(false)}
        title={configDialogData?.title || ''}
        features={configDialogData?.features || []}
      />
    </div>
  );
};

export default SampleViewer; 