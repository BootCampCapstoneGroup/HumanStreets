import React, { useRef, useEffect } from 'react';

function ChatSidebar({ messages, onSendMessage, loading, serverStatus, input, onInputChange, selectedLocation, onClearLocation }) {
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    return (
        <div className="sidebar">
            <header className="sidebar-header">
                <div className="header-title">
                    <h1>Chat</h1>
                    <div className={`status-dot ${serverStatus ? 'online' : 'offline'}`} title={serverStatus ? "Online" : "Offline"}></div>
                </div>
            </header>

            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}`}>
                        <div className="message-content">
                            {msg.content}
                            {msg.isStreaming && <span className="cursor">|</span>}
                        </div>
                        {msg.role === 'assistant' && msg.timeTaken && (
                            <div className="message-meta">
                                Time: {msg.timeTaken}s
                            </div>
                        )}
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>

            {selectedLocation && (
                <div style={{ padding: '8px 15px', background: '#e0f7fa', color: '#006064', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', borderTop: '1px solid #b2ebf2' }}>
                    <span>📍 Pin Dropped: {selectedLocation.latitude.toFixed(4)}, {selectedLocation.longitude.toFixed(4)}</span>
                    <button onClick={onClearLocation} style={{ background: 'transparent', border: 'none', color: '#006064', cursor: 'pointer', fontWeight: 'bold' }}>✕</button>
                </div>
            )}

            <form onSubmit={onSendMessage} className="chat-input-area">
                <input
                    type="text"
                    value={input}
                    onChange={onInputChange}
                    placeholder="Type a message..."
                    disabled={loading}
                    autoFocus
                />
                <button type="submit" disabled={loading}>Send</button>
            </form>
        </div>
    );
}

export default ChatSidebar;
