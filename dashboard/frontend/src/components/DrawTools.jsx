import React from 'react';
import { MousePointer2, Hexagon, Component, MapPin } from 'lucide-react';

const DrawTools = ({ mode, setMode }) => {
    const tools = [
        { id: 'view', icon: MousePointer2, label: 'View', value: 'view' },
        { id: 'point', icon: MapPin, label: 'Point', value: 'drawPoint' },
        { id: 'line', icon: Component, label: 'Line', value: 'drawLineString' }, // Component icon as placeholder for line
        { id: 'polygon', icon: Hexagon, label: 'Polygon', value: 'drawPolygon' },
    ];

    return (
        <div className="absolute top-4 left-4 z-50 bg-white/90 backdrop-blur-sm rounded-lg shadow-lg border border-gray-200 p-2 flex flex-col gap-2">
            {tools.map((tool) => {
                const Icon = tool.icon;
                const isActive = mode === tool.value;
                return (
                    <button
                        key={tool.id}
                        onClick={() => setMode(tool.value)}
                        className={`p-2 rounded-md transition-colors duration-200 flex items-center justify-center ${isActive
                                ? 'bg-blue-600 text-white shadow-sm'
                                : 'text-gray-600 hover:bg-gray-100'
                            }`}
                        title={tool.label}
                    >
                        <Icon size={20} />
                    </button>
                );
            })}
        </div>
    );
};

export default DrawTools;
