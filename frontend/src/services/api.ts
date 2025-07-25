import { SampleData, RandomSampleResponse, PredictionResponse } from '../types';

const API_BASE_URL = 'http://localhost:5001';

export const api = {
  // Get a random sample for teaching
  async getRandomSample(): Promise<RandomSampleResponse> {
    const response = await fetch(`${API_BASE_URL}/random_sample`);
    if (!response.ok) {
      throw new Error('Failed to fetch random sample');
    }
    return response.json();
  },

  // Submit sample for prediction
  async predictSample(sample: SampleData, urlSampleId: number, networkSampleId: number, userSampleId: number): Promise<PredictionResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...sample,
        url_sample_id: urlSampleId,
        network_sample_id: networkSampleId,
        user_sample_id: userSampleId
      }),
    });
    
    if (!response.ok) {
      throw new Error('Failed to get prediction');
    }
    return response.json();
  },

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) {
      throw new Error('Backend is not available');
    }
    return response.json();
  },
}; 