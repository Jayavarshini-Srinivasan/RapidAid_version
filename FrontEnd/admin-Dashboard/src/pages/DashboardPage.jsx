import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, CircularProgress, Card, CardContent } from '@mui/material';
import { DashboardOutlined, LocalHospitalOutlined, Groups2Outlined, EmergencyOutlined } from '@mui/icons-material';
import { adminAPI } from '../services/api';
import '../styles/DashboardPage.css';

export default function DashboardPage() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await adminAPI.getDashboardMetrics();
      setMetrics(response.data.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  const statCards = [
    {
      title: 'Total Drivers',
      value: metrics?.totalDrivers || 0,
      subtitle: `On Duty: ${metrics?.onDutyDrivers || 0}`,
      icon: <Groups2Outlined sx={{ fontSize: 40 }} />,
      color: '#1976d2',
    },
    {
      title: 'Total Patients',
      value: metrics?.totalPatients || 0,
      subtitle: 'Registered',
      icon: <Groups2Outlined sx={{ fontSize: 40 }} />,
      color: '#2e7d32',
    },
    {
      title: 'Active Emergencies',
      value: metrics?.activeEmergencies || 0,
      subtitle: `Total: ${metrics?.totalEmergencies || 0}`,
      icon: <EmergencyOutlined sx={{ fontSize: 40 }} />,
      color: '#d32f2f',
    },
    {
      title: 'Hospitals',
      value: metrics?.totalHospitals || 0,
      subtitle: `Available Beds: ${metrics?.availableBeds || 0}`,
      icon: <LocalHospitalOutlined sx={{ fontSize: 40 }} />,
      color: '#ed6c02',
    },
  ];

  return (
    <Box className="dashboard-container">
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        <DashboardOutlined sx={{ mr: 1, verticalAlign: 'middle' }} />
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {statCards.map((card, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{ height: '100%', bgcolor: card.color, color: 'white' }}>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Box>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {card.value}
                    </Typography>
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      {card.title}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      {card.subtitle}
                    </Typography>
                  </Box>
                  <Box sx={{ opacity: 0.8 }}>{card.icon}</Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}

        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Activity
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {metrics?.completedEmergencies || 0} emergencies completed today
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {metrics?.liveTracking || 0} active live tracking sessions
            </Typography>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
