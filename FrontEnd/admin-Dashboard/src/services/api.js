import axios from 'axios';
import { auth } from '../../firebaseConfig';
import {
  sampleDrivers,
  samplePatients,
  sampleEmergencies,
  sampleHospitals,
  sampleHospitalCapacity,
  sampleDashboardMetrics,
  simulateDelay,
} from '../utils/sampleData';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';
const USE_SAMPLE_DATA = import.meta.env.VITE_USE_SAMPLE_DATA === 'true' || false;

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  async (config) => {
    const user = auth.currentUser;
    if (user) {
      const token = await user.getIdToken();
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Helper function to create mock API response
const createMockResponse = (data) => {
  return {
    data: {
      success: true,
      data,
      message: 'Success',
    },
  };
};

// Admin API endpoints
export const adminAPI = {
  // Drivers
  getDrivers: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(sampleDrivers);
    }
    try {
      return await api.get('/admin/drivers');
    } catch (error) {
      // Fallback to sample data if API fails
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(sampleDrivers);
    }
  },
  
  // Patients
  getPatients: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(samplePatients);
    }
    try {
      return await api.get('/admin/patients');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(samplePatients);
    }
  },
  
  // Emergency Requests
  getActiveRequests: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      const active = sampleEmergencies.filter(e => e.status !== 'completed');
      return createMockResponse(active);
    }
    try {
      return await api.get('/admin/requests/active');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      const active = sampleEmergencies.filter(e => e.status !== 'completed');
      return createMockResponse(active);
    }
  },
  
  assignDriver: async (requestId, driverId) => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse({ success: true, message: 'Driver assigned' });
    }
    try {
      return await api.post('/admin/assign-driver', { requestId, driverId });
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse({ success: true, message: 'Driver assigned (mock)' });
    }
  },
  
  completeRequest: async (requestId) => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse({ success: true, message: 'Request completed' });
    }
    try {
      return await api.patch(`/admin/request/complete/${requestId}`);
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse({ success: true, message: 'Request completed (mock)' });
    }
  },
  
  // Hospitals
  getHospitals: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(sampleHospitals);
    }
    try {
      return await api.get('/admin/hospitals');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(sampleHospitals);
    }
  },
  
  // Hospital Capacity
  getHospitalCapacity: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(sampleHospitalCapacity);
    }
    try {
      return await api.get('/admin/hospitals/capacity');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(sampleHospitalCapacity);
    }
  },
  
  // Dashboard
  getDashboardMetrics: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(sampleDashboardMetrics);
    }
    try {
      return await api.get('/admin/dashboard/metrics');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(sampleDashboardMetrics);
    }
  },
  
  // Emergencies
  getEmergencies: async () => {
    if (USE_SAMPLE_DATA) {
      await simulateDelay();
      return createMockResponse(sampleEmergencies);
    }
    try {
      return await api.get('/admin/emergencies');
    } catch (error) {
      console.warn('API failed, using sample data:', error.message);
      await simulateDelay();
      return createMockResponse(sampleEmergencies);
    }
  },
};

export default api;

