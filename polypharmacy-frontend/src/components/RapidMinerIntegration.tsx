'use client'

import { useState } from 'react'

export function RapidMinerIntegration() {
  const [rapidMinerPath, setRapidMinerPath] = useState('')
  const [status, setStatus] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`])
  }

  const prepareData = async () => {
    setLoading(true)
    addLog('Preparing data for RapidMiner...')
    
    try {
      const response = await fetch('http://localhost:5000/rapidminer/prepare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      const result = await response.json()
      
      if (response.ok) {
        addLog(`Data prepared successfully: ${result.data_file || 'training data'}`)
        if (result.features && Array.isArray(result.features)) {
          addLog(`Features: ${result.features.join(', ')}`)
        }
        if (result.samples) {
          addLog(`Total samples: ${result.samples}`)
        }
        addLog('✅ Data preparation complete!')
        await getStatus()
      } else {
        addLog(`❌ Error: ${result.error || 'Unknown error'}`)
      }
    } catch (error) {
      addLog(`❌ Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const createProcess = async () => {
    setLoading(true)
    addLog('Creating Random Forest process file...')
    
    try {
      const response = await fetch('http://localhost:5000/rapidminer/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      const result = await response.json()
      
      if (response.ok) {
        addLog(`Process file created: ${result.process_file || 'polypharmacy_training.rmp'}`)
        addLog(`Model type: ${result.model_type || 'Random Forest'}`)
        addLog('✅ Process creation complete!')
        await getStatus()
      } else {
        addLog(`❌ Error: ${result.error || 'Unknown error'}`)
      }
    } catch (error) {
      addLog(`❌ Network error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }

  const launchRapidMiner = async () => {
    if (!rapidMinerPath.trim()) {
      addLog('Error: Please specify RapidMiner executable path')
      return
    }

    setLoading(true)
    addLog('Launching RapidMiner...')
    
    try {
      const response = await fetch('http://localhost:5000/rapidminer/launch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rapidminer_path: rapidMinerPath })
      })
      
      const result = await response.json()
      
      if (response.ok) {
        addLog('RapidMiner launched successfully!')
        if (result.process_file) {
          addLog(`Process file: ${result.process_file}`)
        }
        if (result.instructions) {
          addLog('📋 Next steps:')
          result.instructions.forEach((instruction: string) => {
            addLog(`   ${instruction}`)
          })
        }
        addLog('Complete the training in RapidMiner and save the model.')
      } else {
        addLog(`Error: ${result.error}`)
      }
    } catch (error) {
      addLog(`Network error: ${error}`)
    } finally {
      setLoading(false)
    }
  }

  const getStatus = async () => {
    try {
      const response = await fetch('http://localhost:5000/rapidminer/status')
      const result = await response.json()
      setStatus(result)
      addLog('Status updated')
    } catch (error) {
      addLog(`Error getting status: ${error}`)
    }
  }

  const handleCompleteWorkflow = async () => {
    addLog('Starting complete RapidMiner workflow...')
    await prepareData()
    await createProcess()
    await launchRapidMiner()
  }

  const clearLogs = () => {
    setLogs([])
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-semibold text-gray-800">
          RapidMiner Integration
        </h2>
        <button
          onClick={getStatus}
          className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
        >
          Refresh Status
        </button>
      </div>

      {/* Configuration */}
      <div className="mb-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            RapidMiner Executable Path
          </label>
          <input
            type="text"
            value={rapidMinerPath}
            onChange={(e) => setRapidMinerPath(e.target.value)}
            placeholder="C:\Program Files\RapidMiner\RapidMiner Studio\scripts\rapidminer-studio.bat"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="mt-1 text-sm text-gray-500">
            Path to RapidMiner Studio executable or batch file
          </p>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Training Workflow</h3>
        <p className="text-sm text-gray-600 mb-4">Using Random Forest model (optimized for drug interaction prediction)</p>
        
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <button
            onClick={prepareData}
            disabled={loading}
            className="px-4 py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 
                     text-white font-medium rounded-lg transition-colors"
          >
            1. Prepare Data
          </button>
          
          <button
            onClick={createProcess}
            disabled={loading}
            className="px-4 py-3 bg-green-500 hover:bg-green-600 disabled:bg-gray-300 
                     text-white font-medium rounded-lg transition-colors"
          >
            2. Create Process
          </button>
          
          <button
            onClick={launchRapidMiner}
            disabled={loading || !rapidMinerPath.trim()}
            className="px-4 py-3 bg-purple-500 hover:bg-purple-600 disabled:bg-gray-300 
                     text-white font-medium rounded-lg transition-colors"
          >
            3. Launch RapidMiner
          </button>
          
          <button
            onClick={handleCompleteWorkflow}
            disabled={loading || !rapidMinerPath.trim()}
            className="px-4 py-3 bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 
                     text-white font-medium rounded-lg transition-colors"
          >
            Complete Workflow
          </button>
        </div>

        <div className="text-sm text-gray-600 bg-blue-50 p-4 rounded-lg">
          <h4 className="font-medium mb-2">Workflow Steps:</h4>
          <ol className="list-decimal list-inside space-y-1">
            <li>Prepare training data with engineered features</li>
            <li>Create RapidMiner process file for selected model type</li>
            <li>Launch RapidMiner Studio with the prepared process</li>
            <li>Complete training in RapidMiner and export the model</li>
            <li>Load the trained model back into the web application</li>
          </ol>
        </div>
      </div>

      {/* Status Display */}
      {status && (
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-800 mb-2">Current Status</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Data File:</span> 
              <span className={status.data_prepared ? 'text-green-600' : 'text-red-600'}>
                {status.data_prepared ? ' Ready' : ' Not prepared'}
              </span>
            </div>
            <div>
              <span className="font-medium">Process File:</span>
              <span className={status.process_created ? 'text-green-600' : 'text-red-600'}>
                {status.process_created ? ' Ready' : ' Not created'}
              </span>
            </div>
            <div>
              <span className="font-medium">Last Updated:</span> {status.last_updated}
            </div>
            <div>
              <span className="font-medium">Ready for Training:</span>
              <span className={status.ready_for_training ? 'text-green-600' : 'text-orange-600'}>
                {status.ready_for_training ? ' Yes' : ' No'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Activity Logs */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">Activity Logs</h3>
          <button
            onClick={clearLogs}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded"
          >
            Clear
          </button>
        </div>
        
        <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm max-h-60 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-gray-500">No activity logs yet...</p>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="mb-1">
                {log}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-lg flex items-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-4"></div>
            <span className="text-lg">Processing...</span>
          </div>
        </div>
      )}
    </div>
  )
}
