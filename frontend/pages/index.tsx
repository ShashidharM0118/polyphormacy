import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Box,
  Button,
  TextField,
  Autocomplete,
  Alert,
  CircularProgress,
  Chip,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Tabs,
  Tab,
  AppBar,
  Toolbar,
} from '@mui/material';
import {
  PredictionOutlined,
  LocalPharmacyOutlined,
  AnalyticsOutlined,
  WarningAmberOutlined,
  CheckCircleOutlined,
  ErrorOutlined,
} from '@mui/icons-material';
import axios from 'axios';

interface DrugInfo {
  id: string;
  interactions: Array<{
    partner: string;
    side_effect: string;
    severity: string;
    system: string;
  }>;
  side_effects: string[];
  partners: string[];
  unique_side_effects: string[];
  interaction_count: number;
}

interface PredictionResult {
  drug1: string;
  drug2: string;
  prediction: number;
  confidence: number;
  probability_no_interaction: number;
  probability_interaction: number;
  model_used: string;
  known_interactions: Array<{
    partner: string;
    side_effect: string;
    severity: string;
    system: string;
  }>;
  prediction_timestamp: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function Home() {
  const [tabValue, setTabValue] = useState(0);
  const [drugs, setDrugs] = useState<string[]>([]);
  const [selectedDrug1, setSelectedDrug1] = useState<string | null>(null);
  const [selectedDrug2, setSelectedDrug2] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [drugInfo, setDrugInfo] = useState<DrugInfo | null>(null);
  const [drugInfoDialog, setDrugInfoDialog] = useState(false);
  const [modelPerformance, setModelPerformance] = useState<any>(null);

  useEffect(() => {
    fetchDrugs();
    fetchModelPerformance();
  }, []);

  const fetchDrugs = async () => {
    try {
      const response = await axios.get('/api/drugs');
      setDrugs(response.data.drugs);
    } catch (error) {
      console.error('Error fetching drugs:', error);
      setError('Failed to load drug database');
    }
  };

  const fetchModelPerformance = async () => {
    try {
      const response = await axios.get('/api/models');
      setModelPerformance(response.data);
    } catch (error) {
      console.error('Error fetching model performance:', error);
    }
  };

  const handlePredict = async () => {
    if (!selectedDrug1 || !selectedDrug2) {
      setError('Please select both drugs');
      return;
    }

    if (selectedDrug1 === selectedDrug2) {
      setError('Please select different drugs');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/api/predict', {
        drug1: selectedDrug1,
        drug2: selectedDrug2,
        model: 'random_forest'
      });

      setPrediction(response.data);
    } catch (error: any) {
      setError(error.response?.data?.error || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDrugInfo = async (drugId: string) => {
    try {
      const response = await axios.get(`/api/drug/${drugId}`);
      setDrugInfo(response.data);
      setDrugInfoDialog(true);
    } catch (error) {
      setError('Failed to load drug information');
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getRiskLevel = (prediction: PredictionResult) => {
    if (prediction.prediction === 0) return 'No Interaction';
    
    const confidence = prediction.confidence;
    const hasKnownInteractions = prediction.known_interactions.length > 0;
    
    if (hasKnownInteractions) {
      const severities = prediction.known_interactions.map(i => i.severity);
      if (severities.includes('severe')) return 'High Risk';
      if (severities.includes('moderate')) return 'Medium Risk';
      return 'Low Risk';
    }
    
    if (confidence > 0.8) return 'High Risk';
    if (confidence > 0.6) return 'Medium Risk';
    return 'Low Risk';
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'High Risk': return 'error';
      case 'Medium Risk': return 'warning';
      case 'Low Risk': return 'info';
      case 'No Interaction': return 'success';
      default: return 'default';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'High Risk': return <ErrorOutlined />;
      case 'Medium Risk': return <WarningAmberOutlined />;
      case 'Low Risk': return <WarningAmberOutlined />;
      case 'No Interaction': return <CheckCircleOutlined />;
      default: return null;
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" color="primary">
        <Toolbar>
          <LocalPharmacyOutlined sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Polypharmacy Prediction System
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="basic tabs example">
            <Tab icon={<PredictionOutlined />} label="Drug Interaction Prediction" />
            <Tab icon={<AnalyticsOutlined />} label="Model Performance" />
            <Tab icon={<LocalPharmacyOutlined />} label="Drug Database" />
          </Tabs>
        </Box>

        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" gutterBottom>
                    Drug Interaction Prediction
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Select two drugs to predict potential interactions and side effects.
                  </Typography>

                  <Grid container spacing={3} sx={{ mt: 2 }}>
                    <Grid item xs={12} md={6}>
                      <Autocomplete
                        options={drugs}
                        value={selectedDrug1}
                        onChange={(event, newValue) => setSelectedDrug1(newValue)}
                        renderInput={(params) => (
                          <TextField {...params} label="First Drug" variant="outlined" />
                        )}
                      />
                    </Grid>
                    <Grid item xs={12} md={6}>
                      <Autocomplete
                        options={drugs}
                        value={selectedDrug2}
                        onChange={(event, newValue) => setSelectedDrug2(newValue)}
                        renderInput={(params) => (
                          <TextField {...params} label="Second Drug" variant="outlined" />
                        )}
                      />
                    </Grid>
                  </Grid>

                  <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                    <Button
                      variant="contained"
                      onClick={handlePredict}
                      disabled={loading || !selectedDrug1 || !selectedDrug2}
                      startIcon={loading ? <CircularProgress size={20} /> : <PredictionOutlined />}
                    >
                      {loading ? 'Predicting...' : 'Predict Interaction'}
                    </Button>
                    
                    {selectedDrug1 && (
                      <Button
                        variant="outlined"
                        onClick={() => handleDrugInfo(selectedDrug1)}
                      >
                        Info: {selectedDrug1}
                      </Button>
                    )}
                    
                    {selectedDrug2 && (
                      <Button
                        variant="outlined"
                        onClick={() => handleDrugInfo(selectedDrug2)}
                      >
                        Info: {selectedDrug2}
                      </Button>
                    )}
                  </Box>

                  {error && (
                    <Alert severity="error" sx={{ mt: 2 }}>
                      {error}
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {prediction && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Prediction Results
                    </Typography>
                    
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            Drug Combination
                          </Typography>
                          <Typography variant="body1">
                            {prediction.drug1} + {prediction.drug2}
                          </Typography>
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
                            {getRiskIcon(getRiskLevel(prediction))}
                            <Typography variant="h6" sx={{ ml: 1 }}>
                              Risk Level
                            </Typography>
                          </Box>
                          <Chip
                            label={getRiskLevel(prediction)}
                            color={getRiskColor(getRiskLevel(prediction)) as any}
                            size="large"
                          />
                        </Paper>
                      </Grid>
                      
                      <Grid item xs={12} md={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h6" color="primary">
                            Confidence
                          </Typography>
                          <Typography variant="h4" color="text.primary">
                            {(prediction.confidence * 100).toFixed(1)}%
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>

                    <Box sx={{ mt: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        Prediction Details
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2">
                            <strong>Probability of No Interaction:</strong> {(prediction.probability_no_interaction * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2">
                            <strong>Probability of Interaction:</strong> {(prediction.probability_interaction * 100).toFixed(1)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2">
                            <strong>Model Used:</strong> {prediction.model_used}
                          </Typography>
                        </Grid>
                        <Grid item xs={12} md={6}>
                          <Typography variant="body2">
                            <strong>Prediction Time:</strong> {new Date(prediction.prediction_timestamp).toLocaleString()}
                          </Typography>
                        </Grid>
                      </Grid>
                    </Box>

                    {prediction.known_interactions.length > 0 && (
                      <Box sx={{ mt: 3 }}>
                        <Typography variant="h6" gutterBottom>
                          Known Interactions from Database
                        </Typography>
                        <TableContainer component={Paper}>
                          <Table>
                            <TableHead>
                              <TableRow>
                                <TableCell>Side Effect</TableCell>
                                <TableCell>Severity</TableCell>
                                <TableCell>Affected System</TableCell>
                              </TableRow>
                            </TableHead>
                            <TableBody>
                              {prediction.known_interactions.map((interaction, index) => (
                                <TableRow key={index}>
                                  <TableCell>{interaction.side_effect}</TableCell>
                                  <TableCell>
                                    <Chip
                                      label={interaction.severity}
                                      size="small"
                                      color={
                                        interaction.severity === 'severe' ? 'error' :
                                        interaction.severity === 'moderate' ? 'warning' : 'default'
                                      }
                                    />
                                  </TableCell>
                                  <TableCell>{interaction.system}</TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Model Performance Metrics
              </Typography>
              {modelPerformance ? (
                <Grid container spacing={3}>
                  {modelPerformance.binary && (
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom>
                        Binary Classification Models (Interaction vs No Interaction)
                      </Typography>
                      <TableContainer component={Paper}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>Model</TableCell>
                              <TableCell>AUC Score</TableCell>
                              <TableCell>F1 Score</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(modelPerformance.binary).map(([model, metrics]: [string, any]) => (
                              <TableRow key={model}>
                                <TableCell>{model}</TableCell>
                                <TableCell>{metrics.auc_score?.toFixed(3) || 'N/A'}</TableCell>
                                <TableCell>{metrics.f1_score?.toFixed(3) || 'N/A'}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>
                  )}
                  
                  {modelPerformance.multiclass && (
                    <Grid item xs={12}>
                      <Typography variant="h6" gutterBottom>
                        Multiclass Models (Specific Side Effects)
                      </Typography>
                      <TableContainer component={Paper}>
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell>Model Configuration</TableCell>
                              <TableCell>Feature Set</TableCell>
                              <TableCell>Target</TableCell>
                              <TableCell>F1 Score</TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {Object.entries(modelPerformance.multiclass).map(([config, metrics]: [string, any]) => (
                              <TableRow key={config}>
                                <TableCell>{config}</TableCell>
                                <TableCell>{metrics.feature_set}</TableCell>
                                <TableCell>{metrics.target}</TableCell>
                                <TableCell>{metrics.f1_score?.toFixed(3) || 'N/A'}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Grid>
                  )}
                </Grid>
              ) : (
                <Typography>Loading model performance data...</Typography>
              )}
            </CardContent>
          </Card>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Card>
            <CardContent>
              <Typography variant="h5" gutterBottom>
                Drug Database
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Total drugs in database: {drugs.length}
              </Typography>
              
              <Grid container spacing={2}>
                {drugs.slice(0, 20).map((drug) => (
                  <Grid item xs={12} sm={6} md={4} key={drug}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="body1">{drug}</Typography>
                      <Button
                        size="small"
                        onClick={() => handleDrugInfo(drug)}
                        sx={{ mt: 1 }}
                      >
                        View Details
                      </Button>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
              
              {drugs.length > 20 && (
                <Typography variant="body2" sx={{ mt: 2 }}>
                  Showing first 20 drugs. Use the prediction tab to search for specific drugs.
                </Typography>
              )}
            </CardContent>
          </Card>
        </TabPanel>
      </Container>

      {/* Drug Info Dialog */}
      <Dialog
        open={drugInfoDialog}
        onClose={() => setDrugInfoDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Drug Information</DialogTitle>
        <DialogContent>
          {drugInfo && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="h6">{drugInfo.id}</Typography>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Interaction Statistics
                  </Typography>
                  <Typography variant="body2">
                    Total Interactions: {drugInfo.interaction_count}
                  </Typography>
                  <Typography variant="body2">
                    Unique Partners: {drugInfo.partners.length}
                  </Typography>
                  <Typography variant="body2">
                    Unique Side Effects: {drugInfo.unique_side_effects.length}
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Common Side Effects
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {drugInfo.unique_side_effects.slice(0, 10).map((effect, index) => (
                      <Chip key={index} label={effect} size="small" />
                    ))}
                  </Box>
                </Paper>
              </Grid>
              
              {drugInfo.interactions.length > 0 && (
                <Grid item xs={12}>
                  <Typography variant="subtitle1" gutterBottom>
                    Recent Interactions (First 10)
                  </Typography>
                  <TableContainer component={Paper}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Partner Drug</TableCell>
                          <TableCell>Side Effect</TableCell>
                          <TableCell>Severity</TableCell>
                          <TableCell>System</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {drugInfo.interactions.slice(0, 10).map((interaction, index) => (
                          <TableRow key={index}>
                            <TableCell>{interaction.partner}</TableCell>
                            <TableCell>{interaction.side_effect}</TableCell>
                            <TableCell>
                              <Chip
                                label={interaction.severity}
                                size="small"
                                color={
                                  interaction.severity === 'severe' ? 'error' :
                                  interaction.severity === 'moderate' ? 'warning' : 'default'
                                }
                              />
                            </TableCell>
                            <TableCell>{interaction.system}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDrugInfoDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
