import { useState, useEffect } from 'react'
import './index.css'
import ChatSidebar from './components/ChatSidebar'
import MapContainer from './components/MapContainer'
import 'maplibre-gl/dist/maplibre-gl.css';

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you today?', timeTaken: null }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverStatus, setServerStatus] = useState(false)

  const [selectedLocation, setSelectedLocation] = useState(null)

  // Poll for server status
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch('http://localhost:8000/health')
        setServerStatus(res.ok)
      } catch (e) {
        setServerStatus(false)
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 5000)
    return () => clearInterval(interval)
  }, [])

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    const currentInput = input
    setInput('')
    setLoading(true)

    setMessages(prev => [...prev, { role: 'assistant', content: '', timeTaken: null, isStreaming: true }])

    const startTime = Date.now()

    try {
      const payload = { message: currentInput }
      if (selectedLocation) {
        payload.latitude = selectedLocation.latitude
        payload.longitude = selectedLocation.longitude
      }

      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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
      console.error('Error:', error)
      setMessages(prev => {
        const newMessages = [...prev]
        newMessages[newMessages.length - 1].content = "Error: Could not fetch response."
        newMessages[newMessages.length - 1].isStreaming = false
        return newMessages
      })
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
      />
      <MapContainer 
        selectedLocation={selectedLocation}
        onLocationSelect={setSelectedLocation}
      />
    </div>
  )
}

export default App
