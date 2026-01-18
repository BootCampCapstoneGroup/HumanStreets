import React, { useRef, useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import ReactMarkdown from 'react-markdown';

function ChatSidebar({ messages, onSendMessage, loading, serverStatus, input, onInputChange, selectedLocation, onClearLocation, selectedModel, onModelChange, showDebugTokens, onToggleDebugTokens, onStop, theme, onToggleTheme }) {
    const messagesEndRef = useRef(null);
    const [showSettings, setShowSettings] = useState(false);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const renderMessageContent = (content) => {
        const chartRegex = /\[\[CHART:?\s*({[\s\S]*?})\s*\]\]+/i;
        const parts = [];
        let remaining = content;
        let match;
        let iterations = 0;

        while ((match = chartRegex.exec(remaining)) !== null && iterations < 10) {
            iterations++;
            const [fullMatch, jsonStr] = match;
            const index = match.index;

            if (index > 0) {
                let textPart = remaining.substring(0, index);
                if (!showDebugTokens) {
                    textPart = textPart.replace(/\[\[?SHOW_LAYER:\s*\w+\]?\]?/gi, '')
                        .replace(/\[\[?VIZ:\s*{.*?}\]?\]?/gs, '')
                        .replace(/\[\[DEBUG:[\s\S]*?\]\]/gi, '');
                }
                parts.push(
                    <div key={`text-${parts.length}`} className="markdown-body" style={{ fontSize: '0.9rem', lineHeight: '1.5', color: 'var(--sidebar-text-primary)' }}>
                        <ReactMarkdown>{textPart}</ReactMarkdown>
                    </div>
                );
            }

            let chartData = null;
            let parseError = null;
            try {
                chartData = JSON.parse(jsonStr);
            } catch (e) {
                parseError = e;
                console.warn("JSON Parse Error:", e);
            }

            if (chartData) {
                if (showDebugTokens) {
                    parts.push(
                        <div key={`debug-${parts.length}`} style={{ color: 'orange', fontSize: '0.8rem', fontFamily: 'monospace', margin: '5px 0', whiteSpace: 'pre-wrap', border: '1px dashed orange', padding: '5px' }}>
                            DEBUG TOKEN: {fullMatch}
                        </div>
                    );
                }

                parts.push(
                    <div key={`chart-${parts.length}`} className="inline-chart" style={{ margin: '10px 0', width: '100%', minHeight: '300px' }}>
                        <Plot
                            data={chartData.data}
                            layout={{
                                ...chartData.layout,
                                width: undefined,
                                autosize: true,
                                paper_bgcolor: 'rgba(0,0,0,0)',
                                plot_bgcolor: 'rgba(0,0,0,0)',
                                // Contrast Theme Logic:
                                // theme='light' -> Map Light -> Sidebar Dark -> Text White
                                // theme='dark' -> Map Dark -> Sidebar Light -> Text Dark
                                font: { color: theme === 'light' ? '#ffffff' : '#333333' },
                                margin: { l: 40, r: 20, t: 40, b: 40 },
                                legend: { orientation: 'h', y: -0.2 }
                            }}
                            config={{ responsive: true, displayModeBar: false }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler={true}
                        />
                    </div>
                );
            } else {
                parts.push(
                    <div key={`err-${parts.length}`} style={{ padding: '10px', background: 'rgba(231, 76, 60, 0.2)', border: '1px solid var(--error-color)', borderRadius: '5px', margin: '5px 0', fontSize: '0.85rem', color: 'var(--sidebar-text-primary)' }}>
                        <strong>⚠️ Chart Rendering Failed</strong>
                        <br />
                        <span style={{ opacity: 0.8 }}>The agent generated invalid JSON data.</span>
                        {showDebugTokens && (
                            <pre style={{ overflowX: 'auto', marginTop: '5px', fontSize: '0.75rem', color: 'var(--error-color)' }}>{parseError?.message}</pre>
                        )}
                    </div>
                );
            }
            remaining = remaining.substring(index + fullMatch.length);
        }

        if (remaining) {
            let textPart = remaining;
            const partialChartMatch = textPart.match(/\[\[CHART:?[\s\S]*$/i);
            const isStreamingChart = partialChartMatch && loading;

            if (isStreamingChart) {
                textPart = textPart.substring(0, partialChartMatch.index);
            }

            if (!showDebugTokens) {
                textPart = textPart.replace(/\[\[?SHOW_LAYER:\s*\w+\]?\]?/gi, '')
                    .replace(/\[\[?VIZ:\s*{.*?}\]?\]?/gs, '')
                    .replace(/\[\[DEBUG:[\s\S]*?\]\]/gi, '');
            }

            if (textPart) {
                parts.push(
                    <div key={`text-end`} className="markdown-body" style={{ fontSize: '0.9rem', lineHeight: '1.5', color: 'var(--sidebar-text-primary)' }}>
                        <ReactMarkdown>{textPart}</ReactMarkdown>
                    </div>
                );
            }

            if (isStreamingChart) {
                parts.push(
                    <div key="chart-loading" style={{
                        padding: '10px',
                        background: 'rgba(74, 108, 247, 0.1)',
                        border: '1px dashed var(--accent-color)',
                        borderRadius: '8px',
                        margin: '10px 0',
                        color: 'var(--accent-color)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px',
                        fontSize: '0.9rem'
                    }}>
                        <span className="sc-spinner">🔄</span>
                        <em>🎨 Generating Chart (bar)...</em>
                    </div>
                );
            }
        }

        return parts;
    };

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
                            {renderMessageContent(msg.content)}
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
                <div style={{ padding: '8px 15px', background: 'var(--sidebar-bg-secondary)', color: 'var(--sidebar-text-primary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', borderTop: '1px solid var(--sidebar-border)' }}>
                    <span>📍 Pin: {selectedLocation.latitude.toFixed(4)}, {selectedLocation.longitude.toFixed(4)}</span>
                    <button onClick={onClearLocation} style={{ background: 'transparent', border: 'none', color: 'var(--sidebar-text-primary)', cursor: 'pointer', fontWeight: 'bold' }}>✕</button>
                </div>
            )}

            {/* Settings Toggle Area */}
            <div style={{ background: 'var(--sidebar-bg-secondary)', borderTop: '1px solid var(--sidebar-border)' }}>
                <button
                    onClick={() => setShowSettings(!showSettings)}
                    style={{
                        width: '100%',
                        background: 'transparent',
                        border: 'none',
                        color: 'var(--sidebar-text-secondary)',
                        cursor: 'pointer',
                        padding: '2px',
                        fontSize: '12px'
                    }}
                    title="Toggle Settings"
                >
                    {showSettings ? '▼' : '▲'}
                </button>

                {showSettings && (
                    <div style={{ padding: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                        {/* Theme Toggle */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                            <label style={{ color: 'var(--sidebar-text-secondary)' }}>Theme:</label>
                            <button
                                onClick={onToggleTheme}
                                style={{
                                    background: 'var(--sidebar-bg)',
                                    color: 'var(--sidebar-text-primary)',
                                    border: '1px solid var(--sidebar-border)',
                                    borderRadius: '4px',
                                    padding: '4px 8px',
                                    cursor: 'pointer'
                                }}
                            >
                                {theme === 'dark' ? 'Use Light Map' : 'Use Dark Map'}
                                {/* Text reflects action or state? State: Current is Dark. Action: Switch to Light. */}
                                {/* Maybe just "Map: Dark/Light" */}
                            </button>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.9rem' }}>
                            <label style={{ color: 'var(--sidebar-text-secondary)' }}>Model:</label>
                            <select
                                value={selectedModel}
                                onChange={(e) => onModelChange(e.target.value)}
                                style={{
                                    background: 'var(--sidebar-input-bg)',
                                    color: 'var(--sidebar-text-primary)',
                                    border: '1px solid var(--sidebar-border)',
                                    borderRadius: '4px',
                                    padding: '4px',
                                    maxWidth: '120px'
                                }}
                            >
                                <option value="local">Local (LiquidAI)</option>
                                <option value="gemini">Gemini (Free)</option>
                                <option value="openrouter_free">Llama 3.2</option>
                                <option value="deepseek_free">DeepSeek R1</option>
                            </select>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9rem' }}>
                            <input
                                type="checkbox"
                                id="debugTokens"
                                checked={showDebugTokens}
                                onChange={(e) => onToggleDebugTokens(e.target.checked)}
                                style={{ accentColor: 'var(--accent-color)' }}
                            />
                            <label htmlFor="debugTokens" style={{ color: 'var(--sidebar-text-secondary)', fontSize: '0.85rem' }}>Show Debug Tokens</label>
                        </div>

                        <button
                            onClick={() => {
                                const text = JSON.stringify(messages, null, 2);
                                navigator.clipboard.writeText(text).then(() => alert("Conversation copied!"));
                            }}
                            style={{
                                background: 'var(--sidebar-bg)', border: '1px solid var(--sidebar-border)', borderRadius: '4px',
                                color: 'var(--sidebar-text-primary)', padding: '5px', cursor: 'pointer', fontSize: '0.85rem'
                            }}
                        >
                            📋 Copy Conversation
                        </button>
                    </div>
                )}
            </div>

            {/* Example Prompts */}
            <div style={{ padding: '0 10px 10px', display: 'flex', gap: '8px', overflowX: 'auto', whiteSpace: 'nowrap', scrollbarWidth: 'none' }}>
                {[
                    "Show population bar chart",
                    "Turn on walkability heatmap",
                    "Top 5 walkable neighborhoods",
                    "How many scores > 80?",
                    "Show districts layer"
                ].map((prompt, i) => (
                    <button
                        key={i}
                        onClick={() => {
                            if (!loading) {
                                onInputChange({ target: { value: prompt } });
                                setTimeout(() => {
                                    onSendMessage({ preventDefault: () => { } });
                                }, 100);
                            }
                        }}
                        style={{
                            background: 'var(--sidebar-bg-secondary)',
                            border: '1px solid var(--sidebar-border)',
                            borderRadius: '15px',
                            padding: '5px 12px',
                            color: 'var(--sidebar-text-primary)',
                            fontSize: '0.75rem',
                            cursor: 'pointer',
                            flexShrink: 0,
                            transition: 'background 0.2s'
                        }}
                        onMouseOver={(e) => e.target.style.background = 'var(--sidebar-bg)'}
                        onMouseOut={(e) => e.target.style.background = 'var(--sidebar-bg-secondary)'}
                    >
                        {prompt}
                    </button>
                ))}
            </div>

            <form onSubmit={loading ? (e) => { e.preventDefault(); onStop(); } : onSendMessage} className="chat-input-area" style={{ borderTop: 'none', paddingTop: 0 }}>
                <input
                    type="text"
                    value={input}
                    onChange={onInputChange}
                    placeholder="Type a message..."
                    disabled={loading}
                    autoFocus
                />
                <button
                    type="submit"
                    style={{
                        background: loading ? 'var(--error-color)' : 'var(--accent-color)',
                        cursor: 'pointer'
                    }}
                >
                    {loading ? 'Stop' : 'Send'}
                </button>
            </form>
        </div>
    );
}

export default ChatSidebar;
