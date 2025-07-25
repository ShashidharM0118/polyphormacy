'use client'

import { useState, useEffect, useRef } from 'react'

interface Drug {
  id: string
  label: string
}

interface DrugSearchProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
}

export function DrugSearch({ label, value, onChange, placeholder }: DrugSearchProps) {
  const [suggestions, setSuggestions] = useState<Drug[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (value.length >= 2) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      timeoutRef.current = setTimeout(() => {
        fetchSuggestions(value)
      }, 300)
    } else {
      setSuggestions([])
      setShowSuggestions(false)
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [value])

  const fetchSuggestions = async (query: string) => {
    setLoading(true)
    try {
      const response = await fetch(`http://localhost:5000/suggest?q=${encodeURIComponent(query)}&limit=10`)
      const data = await response.json()
      
      if (response.ok) {
        setSuggestions(data.suggestions || [])
        setShowSuggestions(true)
      }
    } catch (error) {
      console.error('Failed to fetch suggestions:', error)
      setSuggestions([])
    } finally {
      setLoading(false)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value
    onChange(newValue)
  }

  const handleSuggestionClick = (drug: Drug) => {
    onChange(drug.id)
    setShowSuggestions(false)
    setSuggestions([])
  }

  const handleInputBlur = () => {
    // Delay hiding suggestions to allow click events
    setTimeout(() => {
      setShowSuggestions(false)
    }, 200)
  }

  const isValidStitchId = (id: string) => {
    return /^CID\d+$/.test(id)
  }

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onFocus={() => {
            if (suggestions.length > 0) {
              setShowSuggestions(true)
            }
          }}
          placeholder={placeholder}
          className={`w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 
                     focus:border-blue-500 outline-none transition-colors ${
                       value && !isValidStitchId(value)
                         ? 'border-red-300 bg-red-50'
                         : 'border-gray-300'
                     }`}
        />
        
        {loading && (
          <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
          </div>
        )}
      </div>

      {/* Validation Message */}
      {value && !isValidStitchId(value) && (
        <p className="mt-1 text-sm text-red-600">
          Please enter a valid STITCH ID (format: CID followed by numbers)
        </p>
      )}

      {/* Suggestions Dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {suggestions.map((drug, index) => (
            <div
              key={drug.id}
              onClick={() => handleSuggestionClick(drug)}
              className="px-4 py-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
            >
              <div className="font-medium text-gray-900">{drug.id}</div>
            </div>
          ))}
        </div>
      )}

      {/* No suggestions message */}
      {showSuggestions && suggestions.length === 0 && value.length >= 2 && !loading && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg">
          <div className="px-4 py-3 text-gray-500 text-center">
            No matching STITCH IDs found
          </div>
        </div>
      )}
    </div>
  )
}
