import React from 'react';

function MapControls({ showNeighborhoods, onToggleNeighborhoods, showWalkability, onToggleWalkability }) {
    return (
        <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'rgba(30, 30, 30, 0.9)',
            padding: '15px',
            borderRadius: '8px',
            color: 'white',
            zIndex: 1000,
            boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
            minWidth: '200px'
        }}>
            <h3 style={{ margin: '0 0 10px 0', fontSize: '1rem', borderBottom: '1px solid #555', paddingBottom: '5px' }}>Map Controls</h3>

            {/* Toggles */}
            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer', marginBottom: '8px' }}>
                    <input
                        type="checkbox"
                        checked={showNeighborhoods}
                        onChange={onToggleNeighborhoods}
                        style={{ marginRight: '10px' }}
                    />
                    Riyadh Neighborhoods
                </label>
                <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                    <input
                        type="checkbox"
                        checked={showWalkability}
                        onChange={onToggleWalkability}
                        style={{ marginRight: '10px' }}
                    />
                    Walkability Heatmap
                </label>
            </div>

            {/* Legend */}
            {showWalkability && (
                <div style={{ marginTop: '15px', borderTop: '1px solid #555', paddingTop: '10px' }}>
                    <div style={{ fontSize: '0.85rem', marginBottom: '5px', fontWeight: 'bold' }}>Walkability Score</div>
                    <div style={{
                        height: '10px',
                        width: '100%',
                        background: 'linear-gradient(to right, red, yellow, green)',
                        borderRadius: '5px',
                        marginBottom: '4px'
                    }}></div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#ccc' }}>
                        <span>Low (0)</span>
                        <span>High (100)</span>
                    </div>
                </div>
            )}
        </div>
    );
}

export default MapControls;
