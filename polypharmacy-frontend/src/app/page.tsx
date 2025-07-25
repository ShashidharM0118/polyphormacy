'use client'

import { useState, useEffect } from 'react'
import { DrugSearch } from '@/components/DrugSearch'
import { PredictionResults } from '@/components/PredictionResults'
import { RapidMinerIntegration } from '@/components/RapidMinerIntegration'
import { ModelPerformance } from '@/components/ModelPerformance'

interface PredictionResult {
  drug1: string
  drug2: string
  features: {
    drug_1_interactions: number
    drug_2_interactions: number
    drug_1_side_effects: number
    drug_2_side_effects: number
    common_partners: number
  }
  predictions: {
    [key: string]: {
      prediction: number
      probability?: number
      error?: string
    }
  }
}

export default function Home() {
  const [drug1, setDrug1] = useState('')
  const [drug2, setDrug2] = useState('')
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('predict')
  const [apiStatus, setApiStatus] = useState<any>(null)

  useEffect(() => {
    checkApiStatus()
  }, [])

  const checkApiStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/')
      const data = await response.json()
      setApiStatus(data)
    } catch (err) {
      console.error('API connection failed:', err)
    }
  }

  const handlePredict = async () => {
    if (!drug1 || !drug2) {
      setError('Please enter both STITCH IDs')
      return
    }

    if (drug1 === drug2) {
      setError('Please enter different STITCH IDs')
      return
    }

    setLoading(true)
    setError('')
    setPredictionResult(null)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ drug1, drug2 }),
      })

      const data = await response.json()

      if (response.ok) {
        setPredictionResult(data)
      } else {
        setError(data.error || 'Prediction failed')
      }
    } catch (err) {
      setError('Network error: Please ensure the API server is running')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setDrug1('')
    setDrug2('')
    setPredictionResult(null)
    setError('')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Polypharmacy Prediction System
          </h1>
          <p className="text-lg text-gray-600 mb-4">
            Advanced Drug Interaction Prediction with Machine Learning
          </p>
          
          {/* API Status */}
          {apiStatus && (
            <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-lg">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              API Connected • {apiStatus.models_loaded} models • {apiStatus.drugs_available} drugs
            </div>
          )}
        </div>

        {/* Navigation Tabs */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-lg shadow-sm p-1">
            <button
              onClick={() => setActiveTab('predict')}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'predict'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              Predict Interactions
            </button>
            <button
              onClick={() => setActiveTab('rapidminer')}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'rapidminer'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              RapidMiner Training
            </button>
            <button
              onClick={() => setActiveTab('performance')}
              className={`px-6 py-2 rounded-md font-medium transition-colors ${
                activeTab === 'performance'
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 hover:text-blue-500'
              }`}
            >
              Model Performance
            </button>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'predict' && (
          <div className="max-w-4xl mx-auto">
            {/* Drug Input Section */}
            <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
              <h2 className="text-2xl font-semibold text-gray-800 mb-6">
                Enter Drug STITCH IDs
              </h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <DrugSearch
                  label="First Drug STITCH ID"
                  value={drug1}
                  onChange={setDrug1}
                  placeholder="e.g., CID100000085"
                />
                
                <DrugSearch
                  label="Second Drug STITCH ID"
                  value={drug2}
                  onChange={setDrug2}
                  placeholder="e.g., CID100000298"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex gap-4 mt-6">
                <button
                  onClick={handlePredict}
                  disabled={loading || !drug1 || !drug2}
                  className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 
                           text-white font-medium py-3 px-6 rounded-lg transition-colors
                           flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Predicting...
                    </>
                  ) : (
                    'Predict Interaction'
                  )}
                </button>
                
                <button
                  onClick={handleClear}
                  className="px-6 py-3 border border-gray-300 text-gray-700 
                           hover:bg-gray-50 rounded-lg transition-colors"
                >
                  Clear
                </button>
              </div>

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-red-800">{error}</p>
                </div>
              )}
            </div>

            {/* Results Section */}
            {predictionResult && (
              <PredictionResults result={predictionResult} />
            )}
          </div>
        )}

        {activeTab === 'rapidminer' && (
          <div className="max-w-4xl mx-auto">
            <RapidMinerIntegration />
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="max-w-4xl mx-auto">
            <ModelPerformance />
          </div>
        )}
      </div>
    </div>
  )
}
