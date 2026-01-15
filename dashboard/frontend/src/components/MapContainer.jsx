import React from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import { ScatterplotLayer } from '@deck.gl/layers';

const INITIAL_VIEW_STATE = {
    latitude: 24.7136, // Default to Riyadh
    longitude: 46.6753,
    zoom: 12,
    pitch: 0,
    bearing: 0
};

// Use a public CARTO style to avoid needing an API key for the demo
const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

function MapContainer({ selectedLocation, onLocationSelect }) {
    const layers = [
        selectedLocation && new ScatterplotLayer({
            id: 'selected-location-layer',
            data: [selectedLocation],
            pickable: true,
            opacity: 1,
            stroked: true,
            filled: true,
            radiusScale: 1,
            radiusMinPixels: 8,
            radiusMaxPixels: 20,
            lineWidthMinPixels: 2,
            getPosition: d => [d.longitude, d.latitude],
            getFillColor: [255, 0, 0],
            getLineColor: [255, 255, 255]
        })
    ].filter(Boolean);

    const [viewState, setViewState] = React.useState(INITIAL_VIEW_STATE);

    return (
        <div className="main-content" style={{ position: 'relative' }}>
            <DeckGL
                initialViewState={undefined}
                viewState={viewState}
                onViewStateChange={({ viewState }) => setViewState(viewState)}
                controller={true}
                layers={layers}
                onClick={({ coordinate }) => {
                    if (coordinate) {
                        onLocationSelect({ longitude: coordinate[0], latitude: coordinate[1] });
                    }
                }}
            >
                <Map
                    mapStyle={MAP_STYLE}
                />
            </DeckGL>
        </div>
    );
}

export default MapContainer;
