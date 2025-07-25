'use client'

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

interface PredictionResultsProps {
  result: PredictionResult
}

export function PredictionResults({ result }: PredictionResultsProps) {
  const getModelName = (key: string) => {
    const names: { [key: string]: string } = {
      binary: 'Interaction Risk',
      severity: 'Severity Level',
      system: 'Affected Body System'
    }
    return names[key] || key
  }

  const getModelDescription = (key: string) => {
    const descriptions: { [key: string]: string } = {
      binary: 'Predicts whether the drug combination is likely to cause interactions',
      severity: 'Estimates the severity level of potential side effects',
      system: 'Identifies which body system might be affected by interactions'
    }
    return descriptions[key] || ''
  }

  const getPredictionText = (key: string, prediction: number) => {
    switch (key) {
      case 'binary':
        return prediction === 1 ? 'High Risk' : 'Low Risk'
      case 'severity':
        const severityLevels = ['Mild', 'Moderate', 'Severe', 'Very Severe', 'Critical']
        return severityLevels[prediction] || `Level ${prediction}`
      case 'system':
        const systems = ['Cardiovascular', 'Nervous', 'Digestive', 'Respiratory', 'Other']
        return systems[prediction] || `System ${prediction}`
      default:
        return `Prediction: ${prediction}`
    }
  }

  const getPredictionColor = (key: string, prediction: number) => {
    switch (key) {
      case 'binary':
        return prediction === 1 ? 'text-red-600 bg-red-50' : 'text-green-600 bg-green-50'
      case 'severity':
        const colors = ['text-green-600 bg-green-50', 'text-yellow-600 bg-yellow-50', 
                       'text-orange-600 bg-orange-50', 'text-red-600 bg-red-50', 'text-purple-600 bg-purple-50']
        return colors[prediction] || 'text-gray-600 bg-gray-50'
      default:
        return 'text-blue-600 bg-blue-50'
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">
        Prediction Results
      </h2>

      {/* Drug Pair Info */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-gray-600">Drug 1 STITCH ID</label>
            <p className="text-lg font-mono text-gray-800">{result.drug1}</p>
          </div>
          <div>
            <label className="text-sm font-medium text-gray-600">Drug 2 STITCH ID</label>
            <p className="text-lg font-mono text-gray-800">{result.drug2}</p>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis Features</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <p className="text-2xl font-bold text-blue-600">{result.features.drug_1_interactions}</p>
            <p className="text-sm text-gray-600">Drug 1 Interactions</p>
          </div>
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <p className="text-2xl font-bold text-blue-600">{result.features.drug_2_interactions}</p>
            <p className="text-sm text-gray-600">Drug 2 Interactions</p>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <p className="text-2xl font-bold text-green-600">{result.features.drug_1_side_effects}</p>
            <p className="text-sm text-gray-600">Drug 1 Side Effects</p>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <p className="text-2xl font-bold text-green-600">{result.features.drug_2_side_effects}</p>
            <p className="text-sm text-gray-600">Drug 2 Side Effects</p>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <p className="text-2xl font-bold text-purple-600">{result.features.common_partners}</p>
            <p className="text-sm text-gray-600">Common Partners</p>
          </div>
        </div>
      </div>

      {/* Predictions */}
      <div>
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Predictions</h3>
        <div className="space-y-4">
          {Object.entries(result.predictions).map(([modelKey, prediction]) => (
            <div key={modelKey} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-semibold text-gray-800">{getModelName(modelKey)}</h4>
                {prediction.error ? (
                  <span className="px-3 py-1 text-red-600 bg-red-50 rounded-full text-sm">
                    Error
                  </span>
                ) : (
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getPredictionColor(modelKey, prediction.prediction)}`}>
                    {getPredictionText(modelKey, prediction.prediction)}
                  </span>
                )}
              </div>
              
              <p className="text-sm text-gray-600 mb-3">{getModelDescription(modelKey)}</p>
              
              {prediction.error ? (
                <p className="text-red-600 text-sm">{prediction.error}</p>
              ) : (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">
                    Prediction: {prediction.prediction}
                  </span>
                  {prediction.probability && (
                    <span className="text-sm text-gray-600">
                      Confidence: {(prediction.probability * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-sm text-yellow-800">
          <strong>Disclaimer:</strong> These predictions are for research purposes only and should not be used as a substitute for professional medical advice. Always consult healthcare professionals for medical decisions.
        </p>
      </div>
    </div>
  )
}
