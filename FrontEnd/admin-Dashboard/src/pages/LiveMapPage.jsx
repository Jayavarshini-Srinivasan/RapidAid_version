import React, { useState, useEffect, useCallback } from 'react';
import { GoogleMap, LoadScript, Marker, InfoWindow } from '@react-google-maps/api';
import { Box, Typography, Paper, Chip, Button, CircularProgress } from '@mui/material';
import { adminAPI } from '../services/api';
import { setupSocketListeners, getSocket } from '../services/socket';
import '../styles/LiveMapPage.css';

const GOOGLE_MAPS_API_KEY = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || 'AIzaSyBV0HSC6CPK2w9URvH_FxNXPjBEG52BGcA';

const mapContainerStyle = {
  width: '100%',
  height: '100%',
  minHeight: '80vh',
};

const defaultCenter = {
  lat: 28.6139, // Default to a central location (adjust as needed)
  lng: 77.2090,
};

const mapOptions = {
  zoomControl: true,
  streetViewControl: false,
  mapTypeControl: true,
  fullscreenControl: true,
};

export default function LiveMapPage() {
  const [drivers, setDrivers] = useState([]);
  const [patients, setPatients] = useState([]);
  const [emergencies, setEmergencies] = useState([]);
  const [selectedMarker, setSelectedMarker] = useState(null);
  const [loading, setLoading] = useState(true);
  const [map, setMap] = useState(null);

  // Load initial data
  useEffect(() => {
    loadMapData();
    
    // Setup Socket.IO listeners
    const cleanup = setupSocketListeners({
      onDriverLocationUpdate: (data) => {
        setDrivers(prev => prev.map(driver => 
          driver.id === data.driverId 
            ? { ...driver, currentLocation: { lat: data.lat, lng: data.lng } }
            : driver
        ));
      },
      onPatientLocationUpdate: (data) => {
        setPatients(prev => prev.map(patient => 
          patient.id === data.patientId
            ? { ...patient, location: { lat: data.lat, lng: data.lng } }
            : patient
        ));
      },
      onNewEmergency: (data) => {
        setEmergencies(prev => [...prev, data]);
      },
      onEmergencyStatusUpdate: (data) => {
        setEmergencies(prev => prev.map(emergency =>
          emergency.id === data.id ? { ...emergency, ...data } : emergency
        ));
      },
    });

    return () => {
      cleanup();
    };
  }, []);

  const loadMapData = async () => {
    try {
      setLoading(true);
      const [driversRes, patientsRes, emergenciesRes] = await Promise.all([
        adminAPI.getDrivers(),
        adminAPI.getPatients(),
        adminAPI.getActiveRequests(),
      ]);

      setDrivers(driversRes.data.data || []);
      setPatients(patientsRes.data.data || []);
      setEmergencies(emergenciesRes.data.data || []);
    } catch (error) {
      console.error('Error loading map data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAssignDriver = async (requestId, driverId) => {
    try {
      await adminAPI.assignDriver(requestId, driverId);
      // Update local state for sample data
      setEmergencies(prev => prev.map(emergency =>
        emergency.id === requestId
          ? {
              ...emergency,
              driverId,
              assignedDriver: drivers.find(d => d.id === driverId),
              status: 'accepted',
            }
          : emergency
      ));
      setSelectedMarker(null);
      await loadMapData(); // Reload to update status
    } catch (error) {
      console.error('Error assigning driver:', error);
    }
  };

  const onMapLoad = useCallback((map) => {
    setMap(map);
  }, []);

  const getStatusLabel = (status) => {
    switch (status) {
      case 'available': return 'Available';
      case 'en-route': return 'En Route';
      case 'transporting': return 'Transporting';
      case 'offline': return 'Offline';
      default: return status;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'available': return 'success';
      case 'en-route': return 'warning';
      case 'transporting': return 'error';
      case 'offline': return 'default';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box className="live-map-container">
      <Typography variant="h4" gutterBottom sx={{ mb: 2 }}>
        Live Map
      </Typography>
      
      <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
        <Box display="flex" gap={2} flexWrap="wrap">
          <Chip 
            label={`Active Ambulances: ${drivers.filter(d => {
              const status = d.ambulanceStatus || d.status;
              return status === 'available' || status === 'en-route' || status === 'transporting';
            }).length}`} 
            color="primary" 
          />
          <Chip label={`Active Emergencies: ${emergencies.length}`} color="error" />
          <Chip label={`Available: ${drivers.filter(d => (d.ambulanceStatus || d.status) === 'available').length}`} color="success" />
          <Chip label={`En Route: ${drivers.filter(d => (d.ambulanceStatus || d.status) === 'en-route').length}`} color="warning" />
          <Chip label={`Transporting: ${drivers.filter(d => (d.ambulanceStatus || d.status) === 'transporting').length}`} color="error" />
        </Box>
      </Paper>

      <LoadScript googleMapsApiKey={GOOGLE_MAPS_API_KEY}>
        <Box className="map-wrapper">
          <GoogleMap
            mapContainerStyle={mapContainerStyle}
            center={defaultCenter}
            zoom={12}
            options={mapOptions}
            onLoad={onMapLoad}
          >
            {/* Driver Markers (Ambulances) - Only Active */}
            {drivers
              .filter(driver => {
                const status = driver.ambulanceStatus || driver.status;
                return driver.currentLocation && 
                       (status === 'available' || status === 'en-route' || status === 'transporting');
              })
              .map((driver) => {
                const status = driver.ambulanceStatus || driver.status;
                // Different colors for different statuses
                let iconColor = 'blue';
                if (status === 'available') iconColor = 'green';
                else if (status === 'en-route') iconColor = 'yellow';
                else if (status === 'transporting') iconColor = 'red';
                
                return (
                  <Marker
                    key={`driver-${driver.id}`}
                    position={{
                      lat: driver.currentLocation.lat,
                      lng: driver.currentLocation.lng,
                    }}
                    icon={{
                      url: `http://maps.google.com/mapfiles/ms/icons/${iconColor}-dot.png`,
                    }}
                    onClick={() => setSelectedMarker({ type: 'driver', data: driver })}
                  />
                );
              })}

            {/* Emergency Request Markers (Patients) */}
            {emergencies
              .filter(emergency => emergency.location)
              .map((emergency) => (
                <Marker
                  key={`emergency-${emergency.id}`}
                  position={{
                    lat: emergency.location.lat,
                    lng: emergency.location.lng,
                  }}
                  icon={{
                    url: 'http://maps.google.com/mapfiles/ms/icons/red-dot.png',
                  }}
                  onClick={() => setSelectedMarker({ type: 'emergency', data: emergency })}
                />
              ))}

            {/* Info Window */}
            {selectedMarker && (
              <InfoWindow
                position={
                  selectedMarker.type === 'driver'
                    ? {
                        lat: selectedMarker.data.currentLocation.lat,
                        lng: selectedMarker.data.currentLocation.lng,
                      }
                    : {
                        lat: selectedMarker.data.location.lat,
                        lng: selectedMarker.data.location.lng,
                      }
                }
                onCloseClick={() => setSelectedMarker(null)}
              >
                <Box sx={{ p: 1, minWidth: '200px' }}>
                  {selectedMarker.type === 'driver' ? (
                    <>
                      <Typography variant="h6">{selectedMarker.data.name}</Typography>
                      <Typography variant="body2">Phone: {selectedMarker.data.phone}</Typography>
                      <Typography variant="body2">Ambulance: {selectedMarker.data.ambulanceNumber}</Typography>
                      <Typography variant="body2">Type: {selectedMarker.data.type || 'N/A'}</Typography>
                      <Chip 
                        label={getStatusLabel(selectedMarker.data.ambulanceStatus || selectedMarker.data.status)} 
                        size="small" 
                        color={getStatusColor(selectedMarker.data.ambulanceStatus || selectedMarker.data.status)} 
                        sx={{ mt: 1 }}
                      />
                    </>
                  ) : (
                    <>
                      <Typography variant="h6">Emergency Request</Typography>
                      <Typography variant="body2">Status: {selectedMarker.data.status}</Typography>
                      {selectedMarker.data.assignedDriver && (
                        <Typography variant="body2">
                          Driver: {selectedMarker.data.assignedDriver.name}
                        </Typography>
                      )}
                      {!selectedMarker.data.assignedDriver && drivers.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" display="block" sx={{ mb: 1 }}>
                            Available Drivers:
                          </Typography>
                          {drivers
                            .filter(d => d.status === 'on-duty')
                            .map(driver => (
                              <Button
                                key={driver.id}
                                size="small"
                                variant="contained"
                                onClick={() => handleAssignDriver(selectedMarker.data.id, driver.id)}
                                sx={{ mr: 1, mb: 1 }}
                              >
                                Assign {driver.name}
                              </Button>
                            ))}
                        </Box>
                      )}
                    </>
                  )}
                </Box>
              </InfoWindow>
            )}
          </GoogleMap>
        </Box>
      </LoadScript>
    </Box>
  );
}

