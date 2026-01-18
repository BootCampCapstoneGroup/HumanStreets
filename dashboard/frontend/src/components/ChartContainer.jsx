import React from 'react';
import Plot from 'react-plotly.js';

const ChartContainer = ({ chartConfig, onClose }) => {
    if (!chartConfig) return null;

    return (
        <div className="absolute bottom-4 right-4 z-50 bg-white p-4 rounded-lg shadow-xl border border-gray-200" style={{ maxWidth: '600px', maxHeight: '400px', overflow: 'auto' }}>
            <div className="flex justify-between items-center mb-2">
                <h3 className="font-semibold text-gray-700">Analysis Result</h3>
                <button onClick={onClose} className="text-gray-500 hover:text-gray-700 font-bold">&times;</button>
            </div>
            <Plot
                data={chartConfig.data}
                layout={{
                    ...chartConfig.layout,
                    width: undefined,
                    height: 300,
                    margin: { t: 30, b: 40, l: 50, r: 20 },
                    autosize: true
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
            />
        </div>
    );
};

export default ChartContainer;
