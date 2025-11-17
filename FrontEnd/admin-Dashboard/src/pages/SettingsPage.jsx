import React from 'react';
import { Box, Typography, Paper, TextField, Button, Grid } from '@mui/material';
import { SettingsOutlined } from '@mui/icons-material';
import '../styles/SettingsPage.css';

export default function SettingsPage() {
  return (
    <Box className="settings-container">
      <Typography variant="h4" gutterBottom sx={{ mb: 3, fontWeight: 'bold' }}>
        <SettingsOutlined sx={{ mr: 1, verticalAlign: 'middle' }} />
        Settings
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Settings
            </Typography>
            <TextField
              fullWidth
              label="API Base URL"
              defaultValue="http://localhost:5000/api"
              margin="normal"
            />
            <TextField
              fullWidth
              label="Socket.IO URL"
              defaultValue="http://localhost:5000"
              margin="normal"
            />
            <Button variant="contained" sx={{ mt: 2 }}>
              Save Settings
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Notification Settings
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Configure notification preferences and alerts
            </Typography>
            <Button variant="outlined" sx={{ mt: 2 }}>
              Configure Notifications
            </Button>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}

