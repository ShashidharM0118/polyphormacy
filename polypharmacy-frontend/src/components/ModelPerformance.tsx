'use client'

import { useState, useEffect } from 'react'

interface ModelPerformanceData {
  [modelName: string]: {
    accuracy?: number
    precision?: number
    recall?: number
    f1_score?: number
    classification_report?: string
    confusion_matrix?: number[][]
    feature_importance?: { [feature: string]: number }
  }
}

export function ModelPerformance() {
  const [performance, setPerformance] = useState<ModelPerformanceData>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchPerformance()
  }, [])

  const fetchPerformance = async () => {
    try {
      const response = await fetch('http://localhost:5000/performance')
      const data = await response.json()
      
      if (response.ok) {
        setPerformance(data)
      } else {
        setError(data.error || 'Failed to fetch performance data')
      }
    } catch (err) {
      setError('Network error: Please ensure the API server is running')
    } finally {
      setLoading(false)
    }
  }

  const formatMetric = (value: number) => {
    return (value * 100).toFixed(1) + '%'
  }

  const getMetricColor = (value: number) => {
    if (value >= 0.8) return 'text-green-600'
    if (value >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getModelDisplayName = (modelName: string) => {
    const names: { [key: string]: string } = {
      binary: 'Interaction Risk Model',
      severity: 'Severity Level Model',
      system: 'Body System Model'
    }
    return names[modelName] || modelName
  }

  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-gray-600">Loading performance data...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="text-center py-8">
          <div className="text-red-600 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Error Loading Performance Data</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={fetchPerformance}
            className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-2xl font-semibold text-gray-800 mb-6">
          Model Performance Metrics
        </h2>

        {Object.keys(performance).length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-600">No performance data available</p>
          </div>
        ) : (
          <div className="space-y-8">
            {Object.entries(performance).map(([modelName, metrics]) => (
              <div key={modelName} className="border border-gray-200 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-gray-800 mb-4">
                  {getModelDisplayName(modelName)}
                </h3>

                {/* Core Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  {metrics.accuracy !== undefined && (
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <p className={`text-2xl font-bold ${getMetricColor(metrics.accuracy)}`}>
                        {formatMetric(metrics.accuracy)}
                      </p>
                      <p className="text-sm text-gray-600">Accuracy</p>
                    </div>
                  )}
                  
                  {metrics.precision !== undefined && (
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <p className={`text-2xl font-bold ${getMetricColor(metrics.precision)}`}>
                        {formatMetric(metrics.precision)}
                      </p>
                      <p className="text-sm text-gray-600">Precision</p>
                    </div>
                  )}
                  
                  {metrics.recall !== undefined && (
                    <div className="text-center p-4 bg-yellow-50 rounded-lg">
                      <p className={`text-2xl font-bold ${getMetricColor(metrics.recall)}`}>
                        {formatMetric(metrics.recall)}
                      </p>
                      <p className="text-sm text-gray-600">Recall</p>
                    </div>
                  )}
                  
                  {metrics.f1_score !== undefined && (
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <p className={`text-2xl font-bold ${getMetricColor(metrics.f1_score)}`}>
                        {formatMetric(metrics.f1_score)}
                      </p>
                      <p className="text-sm text-gray-600">F1 Score</p>
                    </div>
                  )}
                </div>

                {/* Feature Importance */}
                {metrics.feature_importance && (
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-gray-800 mb-3">Feature Importance</h4>
                    <div className="space-y-2">
                      {Object.entries(metrics.feature_importance)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([feature, importance]) => (
                          <div key={feature} className="flex items-center">
                            <span className="w-40 text-sm text-gray-600 truncate">{feature}</span>
                            <div className="flex-1 mx-3 bg-gray-200 rounded-full h-2">
                              <div 
                                className="bg-blue-500 h-2 rounded-full" 
                                style={{ width: `${(importance * 100)}%` }}
                              ></div>
                            </div>
                            <span className="text-sm text-gray-800 w-12 text-right">
                              {(importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}

                {/* Confusion Matrix */}
                {metrics.confusion_matrix && (
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-gray-800 mb-3">Confusion Matrix</h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full border border-gray-300">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700">
                              Predicted →
                            </th>
                            {metrics.confusion_matrix[0].map((_, index) => (
                              <th key={index} className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700">
                                Class {index}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {metrics.confusion_matrix.map((row, rowIndex) => (
                            <tr key={rowIndex}>
                              <td className="border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-50">
                                Class {rowIndex}
                              </td>
                              {row.map((value, colIndex) => (
                                <td key={colIndex} className="border border-gray-300 px-4 py-2 text-center">
                                  <span className={rowIndex === colIndex ? 'font-bold text-green-600' : 'text-gray-800'}>
                                    {value}
                                  </span>
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Classification Report */}
                {metrics.classification_report && (
                  <div>
                    <h4 className="text-lg font-semibold text-gray-800 mb-3">Classification Report</h4>
                    <pre className="bg-gray-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto font-mono">
                      {metrics.classification_report}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Performance Summary */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-800 mb-4">Performance Summary</h3>
        
        <div className="space-y-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h4 className="font-medium text-blue-800 mb-2">Model Recommendations</h4>
            <div className="text-sm text-blue-700 space-y-1">
              <p>• The binary interaction model provides the most reliable predictions for general risk assessment</p>
              <p>• Severity level predictions should be used with caution due to class imbalance</p>
              <p>• Feature engineering significantly improves model performance across all tasks</p>
            </div>
          </div>
          
          <div className="p-4 bg-yellow-50 rounded-lg">
            <h4 className="font-medium text-yellow-800 mb-2">Data Quality Notes</h4>
            <div className="text-sm text-yellow-700 space-y-1">
              <p>• Model performance is limited by available training data</p>
              <p>• Consider collecting more diverse drug interaction examples</p>
              <p>• Regular model retraining recommended as new data becomes available</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
