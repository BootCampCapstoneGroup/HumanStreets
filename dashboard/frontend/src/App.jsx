import React, { useState, useEffect } from 'react'
import './colors.css'
import './index.css'
import ChatSidebar from './components/ChatSidebar'
import MapContainer from './components/MapContainer'
import 'maplibre-gl/dist/maplibre-gl.css';
import { API_BASE_URL } from './config';
// ChartContainer removed as per user request

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '# Welcome to HumanStreets AI\n\n**What I can do:**\n- Analyze Riyadh\'s walkability\n- Search for neighborhoods\n- Visualize data on the map\n- Generate Pie & Bar charts\n\n**Try asking:**\n- `Show me Al-Nathim`\n- `Show Al-Narjis and Al-Malqa`\n\n**What to expect:**\nI will query the database, show polygons on the map, and zoom to the results!', timeTaken: null }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState(false)

  // Theme State
  const [theme, setTheme] = useState('light')

  // Apply Theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const [selectedLocation, setSelectedLocation] = useState(null)
  const [drawnFeatures, setDrawnFeatures] = useState({ type: 'FeatureCollection', features: [] })
  const [drawMode, setDrawMode] = useState('view') // view, drawPoint, drawLineString, drawPolygon

  // Handler for map clicks
  const handleMapClick = (event) => {
    if (drawMode === 'view' && event && event.lngLat) {
      setSelectedLocation({
        latitude: event.lngLat.lat,
        longitude: event.lngLat.lng
      })
    }
  }
  const [modelProvider, setModelProvider] = useState('local') // 'local' or 'gemini'

  // Debug State
  const [showDebugTokens, setShowDebugTokens] = useState(false)

  // Map Layers State
  const [showNeighborhoods, setShowNeighborhoods] = useState(false)
  const [showWalkability, setShowWalkability] = useState(false) // H3 Heatmap

  // Zoom Control
  const [zoomToBbox, setZoomToBbox] = useState(null)

  // Agent Visualization State
  const [vizConfig, setVizConfig] = useState(null)

  // Helper to calc bounds
  const calculateBounds = (geojson) => {
    let minLng = 180, minLat = 90, maxLng = -180, maxLat = -90;
    let found = false;
    const features = Array.isArray(geojson) ? geojson : geojson.features;
    if (!features) return null;

    features.forEach(f => {
      const geom = f.geometry;
      if (geom.type === 'Polygon') {
        geom.coordinates[0].forEach(coord => {
          if (Array.isArray(coord)) {
            const [lng, lat] = coord;
            minLng = Math.min(minLng, lng);
            minLat = Math.min(minLat, lat);
            maxLng = Math.max(maxLng, lng);
            maxLat = Math.max(maxLat, lat);
            found = true;
          }
        });
      } else if (geom.type === 'MultiPolygon') {
        geom.coordinates.forEach(poly => {
          poly[0].forEach(coord => {
            const [lng, lat] = coord;
            minLng = Math.min(minLng, lng);
            minLat = Math.min(minLat, lat);
            maxLng = Math.max(maxLng, lng);
            maxLat = Math.max(maxLat, lat);
            found = true;
          });
        });
      }
    });
    return found ? [minLng, minLat, maxLng, maxLat] : null;
  };

  // Poll for server status
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`)
        setServerStatus(res.ok)
      } catch (e) {
        setServerStatus(false)
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  // Ref for aborting fetch
  const abortControllerRef = React.useRef(null)

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setLoading(false)
      setMessages(prev => {
        const newMessages = [...prev]
        const lastMsg = newMessages[newMessages.length - 1]
        // Mark as stopped or just finish streaming
        lastMsg.content += ' [Stopped]'
        lastMsg.isStreaming = false
        return newMessages
      })
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    const currentInput = input
    setInput('')
    setLoading(true)
    // Removed setChartConfig(null)

    setMessages(prev => [...prev, { role: 'assistant', content: '', timeTaken: null, isStreaming: true }])

    const startTime = Date.now()
    abortControllerRef.current = new AbortController()

    try {
      // Construct history (exclude current user message and system messages if any)
      const history = messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const payload = {
        message: currentInput,
        history: history, // Send conversation history
        model_provider: modelProvider,
        drawn_features: drawnFeatures
      }
      if (selectedLocation) {
        payload.latitude = selectedLocation.latitude
        payload.longitude = selectedLocation.longitude
      }

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) throw new Error('Network response was not ok')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let updatedContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value, { stream: true })
        updatedContent += text

        // Check for Layer Triggers (Robust to single/double brackets)
        const layerMatch = updatedContent.match(/\[\[?SHOW_LAYER:\s*(\w+(?:_\w+)?)\]?\]?/i);
        if (layerMatch) {
          const layerName = layerMatch[1].toUpperCase();
          if (layerName === 'NEIGHBORHOODS') {
            if (!showNeighborhoods) setShowNeighborhoods(true);
          } else if (layerName === 'WALKABILITY') {
            if (!showWalkability) setShowWalkability(true);
          } else if (layerName === 'QUERY_RESULT') {
            // Fetch and Append to Drawn Features
            fetch(`${API_BASE_URL}/api/layers/query_result`)
              .then(res => res.json())
              .then(data => {
                if (data && data.features) {
                  setDrawnFeatures(prev => ({
                    ...prev,
                    features: [...prev.features, ...data.features]
                  }));

                  // Auto-Zoom to ALL Features (Existing + New)
                  const allFeatures = [...drawnFeatures.features, ...data.features];
                  const bounds = calculateBounds({ features: allFeatures });

                  if (bounds) {
                    setZoomToBbox(bounds);
                  }
                }
              })
              .catch(e => console.error("Failed to load query result:", e));
          }
        }

        // Removed [[CHART]] parsing block here as we don't need global state for it

        // Check for Structured VIZ Tag: [[VIZ: { ... }]]
        const vizMatch = updatedContent.match(/\[\[VIZ:\s*({.*?})\]\]/s);
        if (vizMatch) {
          try {
            const config = JSON.parse(vizMatch[1]);
            setVizConfig(config);
          } catch (e) {
            console.error("Failed to parse VIZ tag:", e);
          }
        }

        setMessages(prev => {
          const newMessages = [...prev]
          const lastMsg = newMessages[newMessages.length - 1]
          lastMsg.content = updatedContent
          return newMessages
        })
      }

      const endTime = Date.now()
      setMessages(prev => {
        const newMessages = [...prev]
        const lastMsg = newMessages[newMessages.length - 1]
        lastMsg.isStreaming = false
        lastMsg.timeTaken = ((endTime - startTime) / 1000).toFixed(2)
        return newMessages
      })

    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Fetch aborted');
      } else {
        console.error('Error:', error)
        setMessages(prev => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1].content = "Error: Could not fetch response."
          newMessages[newMessages.length - 1].isStreaming = false
          return newMessages
        })
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <ChatSidebar
        messages={messages}
        onSendMessage={sendMessage}
        loading={loading}
        serverStatus={serverStatus}
        input={input}
        onInputChange={(e) => setInput(e.target.value)}
        selectedLocation={selectedLocation}
        onClearLocation={() => setSelectedLocation(null)}
        selectedModel={modelProvider}
        onModelChange={setModelProvider}
        showDebugTokens={showDebugTokens}
        onToggleDebugTokens={setShowDebugTokens}
        onStop={handleStop}
        theme={theme}
        onToggleTheme={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
      />
      <MapContainer
        selectedLocation={selectedLocation}
        onLocationSelect={setSelectedLocation}
        // Layers
        showNeighborhoods={showNeighborhoods}
        onToggleNeighborhoods={() => setShowNeighborhoods(!showNeighborhoods)}
        showWalkability={showWalkability}
        onToggleWalkability={() => setShowWalkability(!showWalkability)}
        // Drawing
        drawnFeatures={drawnFeatures}
        onUpdateFeatures={setDrawnFeatures}
        drawMode={drawMode}
        onSetDrawMode={setDrawMode}
        // Viz Config
        vizConfig={vizConfig}
        showQueryResult={false}
        zoomToBbox={zoomToBbox}
        theme={theme}
      />
    </div>
  )
}

export default App
