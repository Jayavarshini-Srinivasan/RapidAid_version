// Sample data for testing the dashboard without backend

export const sampleDrivers = [
  {
    id: 'd1',
    name: 'John Smith',
    email: 'john.smith@rapidaid.com',
    phone: '+1234567890',
    ambulanceNumber: 'A12',
    status: 'transporting',
    ambulanceStatus: 'transporting',
    currentLocation: {
      lat: 28.6139,
      lng: 77.2090,
    },
    license: 'DL-12345',
    type: 'ALS',
    routeHistory: '12.4 mi',
    lastUpdated: '1 min ago',
  },
  {
    id: 'd2',
    name: 'Sarah Johnson',
    email: 'sarah.j@rapidaid.com',
    phone: '+1234567891',
    ambulanceNumber: 'A09',
    status: 'available',
    ambulanceStatus: 'available',
    currentLocation: {
      lat: 28.6140,
      lng: 77.2091,
    },
    license: 'DL-12346',
    type: 'BLS',
    routeHistory: '0 mi',
    lastUpdated: '2 min ago',
  },
  {
    id: 'd3',
    name: 'Mike Davis',
    email: 'mike.d@rapidaid.com',
    phone: '+1234567892',
    ambulanceNumber: 'A15',
    status: 'en-route',
    ambulanceStatus: 'en-route',
    currentLocation: {
      lat: 28.6145,
      lng: 77.2095,
    },
    license: 'DL-12347',
    type: 'ALS',
    routeHistory: '8.2 mi',
    lastUpdated: '30 sec ago',
  },
  {
    id: 'd4',
    name: 'Emily Chen',
    email: 'emily.c@rapidaid.com',
    phone: '+1234567893',
    ambulanceNumber: 'A21',
    status: 'en-route',
    ambulanceStatus: 'en-route',
    currentLocation: {
      lat: 28.6150,
      lng: 77.2100,
    },
    license: 'DL-12348',
    type: 'BLS',
    routeHistory: '6.7 mi',
    lastUpdated: '45 sec ago',
  },
  {
    id: 'd5',
    name: 'Robert Wilson',
    email: 'robert.w@rapidaid.com',
    phone: '+1234567894',
    ambulanceNumber: 'A08',
    status: 'offline',
    ambulanceStatus: 'offline',
    currentLocation: null,
    license: 'DL-12349',
    type: 'ALS',
    routeHistory: '0 mi',
    lastUpdated: '5 min ago',
  },
];

export const samplePatients = [
  {
    id: 'p1',
    patientId: 'P56',
    name: 'Robert Martinez',
    age: 67,
    email: 'robert.m@email.com',
    phone: '+1987654321',
    emergencyType: 'Cardiac Arrest',
    location: { lat: 40.7580, lng: -73.9855 },
    assignedAmbulance: 'A12',
    pickupStatus: 'en-route',
    priority: 'HIGH',
    callTime: '14:23',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 3600,
    },
  },
  {
    id: 'p2',
    patientId: 'P72',
    name: 'Lisa Anderson',
    age: 34,
    email: 'lisa.a@email.com',
    phone: '+1987654322',
    emergencyType: 'Severe Trauma',
    location: { lat: 40.7570, lng: -73.9000 },
    assignedAmbulance: 'A15',
    pickupStatus: 'picked-up',
    priority: 'HIGH',
    callTime: '14:18',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 1800,
    },
  },
  {
    id: 'p3',
    patientId: 'P45',
    name: 'David Lee',
    age: 45,
    email: 'david.lee@email.com',
    phone: '+1987654323',
    emergencyType: 'Respiratory Distress',
    location: { lat: 40.7550, lng: -73.9800 },
    assignedAmbulance: 'A21',
    pickupStatus: 'en-route',
    priority: 'MEDIUM',
    callTime: '14:15',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 2700,
    },
  },
  {
    id: 'p4',
    patientId: 'P38',
    name: 'Maria Garcia',
    age: 52,
    email: 'maria.g@email.com',
    phone: '+1987654324',
    emergencyType: 'Stroke',
    location: { lat: 40.7500, lng: -73.9700 },
    assignedAmbulance: 'A09',
    pickupStatus: 'awaiting-pickup',
    priority: 'CRITICAL',
    callTime: '14:10',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 3600,
    },
  },
  {
    id: 'p5',
    patientId: 'P61',
    name: 'James Wilson',
    age: 29,
    email: 'james.w@email.com',
    phone: '+1987654325',
    emergencyType: 'Fracture',
    location: { lat: 40.7600, lng: -73.9900 },
    assignedAmbulance: 'A12',
    pickupStatus: 'arrived',
    priority: 'LOW',
    callTime: '13:45',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 7200,
    },
  },
  {
    id: 'p6',
    patientId: 'P49',
    name: 'Sarah Thompson',
    age: 41,
    email: 'sarah.t@email.com',
    phone: '+1987654326',
    emergencyType: 'Chest Pain',
    location: { lat: 40.7520, lng: -73.9750 },
    assignedAmbulance: 'A21',
    pickupStatus: 'en-route',
    priority: 'HIGH',
    callTime: '14:20',
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 1200,
    },
  },
];

export const sampleEmergencies = [
  {
    id: 'e1',
    patientName: 'Robert Brown',
    patientId: 'p1',
    severity: 'High',
    status: 'in_progress',
    driverId: 'd1',
    assignedDriver: {
      name: 'John Smith',
      ambulanceNumber: 'AMB-001',
    },
    location: {
      lat: 28.6135,
      lng: 77.2085,
    },
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 3600, // 1 hour ago
    },
    description: 'Chest pain, difficulty breathing',
  },
  {
    id: 'e2',
    patientName: 'Lisa Anderson',
    patientId: 'p2',
    severity: 'Medium',
    status: 'accepted',
    driverId: 'd2',
    assignedDriver: {
      name: 'Sarah Johnson',
      ambulanceNumber: 'AMB-002',
    },
    location: {
      lat: 28.6145,
      lng: 77.2100,
    },
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 1800, // 30 minutes ago
    },
    description: 'Accident on highway',
  },
  {
    id: 'e3',
    patientName: 'David Lee',
    patientId: 'p3',
    severity: 'Critical',
    status: 'pending',
    driverId: null,
    assignedDriver: null,
    location: {
      lat: 28.6150,
      lng: 77.2105,
    },
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 600, // 10 minutes ago
    },
    description: 'Unconscious, needs immediate attention',
  },
  {
    id: 'e4',
    patientName: 'Maria Garcia',
    patientId: 'p4',
    severity: 'Low',
    status: 'completed',
    driverId: 'd4',
    assignedDriver: {
      name: 'Emily Davis',
      ambulanceNumber: 'AMB-004',
    },
    location: {
      lat: 28.6120,
      lng: 77.2075,
    },
    createdAt: {
      seconds: Math.floor(Date.now() / 1000) - 86400, // 1 day ago
    },
    description: 'Minor injury, treated and discharged',
  },
];

export const sampleHospitals = [
  {
    id: 'h1',
    name: 'City General Hospital',
    address: '123 Main Street, City Center',
    totalBeds: 200,
    availableBeds: 45,
    occupiedBeds: 155,
    occupancyRate: 78,
    location: {
      lat: 28.6200,
      lng: 77.2150,
    },
    wards: {
      icu: { total: 30, occupied: 20, available: 10, occupancyRate: 67 },
      emergency: { total: 50, occupied: 30, available: 20, occupancyRate: 60 },
      general: { total: 80, occupied: 50, available: 30, occupancyRate: 63 },
      trauma: { total: 40, occupied: 30, available: 10, occupancyRate: 75 },
    },
  },
  {
    id: 'h2',
    name: 'Regional Medical Center',
    address: '456 Health Avenue',
    totalBeds: 150,
    availableBeds: 12,
    occupiedBeds: 138,
    occupancyRate: 92,
    location: {
      lat: 28.6100,
      lng: 77.2050,
    },
    wards: {
      icu: { total: 25, occupied: 18, available: 7, occupancyRate: 72 },
      emergency: { total: 40, occupied: 35, available: 5, occupancyRate: 88 },
      general: { total: 60, occupied: 55, available: 5, occupancyRate: 92 },
      trauma: { total: 25, occupied: 20, available: 5, occupancyRate: 80 },
    },
  },
  {
    id: 'h3',
    name: 'Emergency Care Hospital',
    address: '789 Emergency Road',
    totalBeds: 100,
    availableBeds: 8,
    occupiedBeds: 92,
    occupancyRate: 92,
    location: {
      lat: 28.6050,
      lng: 77.2000,
    },
    wards: {
      icu: { total: 20, occupied: 15, available: 5, occupancyRate: 75 },
      emergency: { total: 30, occupied: 28, available: 2, occupancyRate: 93 },
      general: { total: 35, occupied: 32, available: 3, occupancyRate: 91 },
      trauma: { total: 15, occupied: 12, available: 3, occupancyRate: 80 },
    },
  },
  {
    id: 'h4',
    name: 'Community Health Center',
    address: '321 Community Drive',
    totalBeds: 80,
    availableBeds: 25,
    occupiedBeds: 55,
    occupancyRate: 69,
    location: {
      lat: 28.6250,
      lng: 77.2200,
    },
    wards: {
      icu: { total: 15, occupied: 10, available: 5, occupancyRate: 67 },
      emergency: { total: 25, occupied: 15, available: 10, occupancyRate: 60 },
      general: { total: 30, occupied: 20, available: 10, occupancyRate: 67 },
      trauma: { total: 10, occupied: 7, available: 3, occupancyRate: 70 },
    },
  },
];

// Aggregate hospital capacity data
export const sampleHospitalCapacity = {
  totalBeds: 530,
  availableBeds: 90,
  occupiedBeds: 440,
  occupancyRate: 83,
  incoming: 8,
  wards: {
    icu: { total: 90, occupied: 63, available: 27, occupancyRate: 70 },
    emergency: { total: 145, occupied: 108, available: 37, occupancyRate: 74 },
    general: { total: 205, occupied: 157, available: 48, occupancyRate: 77 },
    trauma: { total: 90, occupied: 69, available: 21, occupancyRate: 77 },
  },
};

export const sampleDashboardMetrics = {
  totalDrivers: 4,
  onDutyDrivers: 3,
  totalPatients: 5,
  totalEmergencies: 4,
  activeEmergencies: 3,
  completedEmergencies: 1,
  liveTracking: 3,
  totalHospitals: 4,
  availableBeds: 90,
};

// Helper function to simulate API delay
export const simulateDelay = (ms = 500) => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

