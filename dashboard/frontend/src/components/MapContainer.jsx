import React, { useState, useEffect } from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';
import { ScatterplotLayer, GeoJsonLayer, H3HexagonLayer } from 'deck.gl';
import { EditableGeoJsonLayer } from '@nebula.gl/layers';
import { ViewMode, DrawPointMode, DrawLineStringMode, DrawPolygonMode } from '@nebula.gl/edit-modes';
import { API_BASE_URL } from '../config';
import MapControls from './MapControls';

const INITIAL_VIEW_STATE = {
    latitude: 24.7136, // Default to Riyadh
    longitude: 46.6753,
    zoom: 12,
    pitch: 0,
    bearing: 0
};

// Map Styles
const MAP_STYLE_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const MAP_STYLE_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

function MapContainer({ selectedLocation, onLocationSelect, showNeighborhoods, showWalkability, zoomToBbox, onToggleNeighborhoods, onToggleWalkability, drawnFeatures = { type: 'FeatureCollection', features: [] }, onUpdateFeatures, drawMode, onSetDrawMode, theme }) {
    const [neighborhoodData, setNeighborhoodData] = useState(null);
    const [h3Data, setH3Data] = useState(null);
    const [selectedFeatureIndexes, setSelectedFeatureIndexes] = useState([]);

    // Fetch Neighborhoods
    useEffect(() => {
        if (!neighborhoodData) {
            fetch(`${API_BASE_URL}/api/layers/neighborhoods`)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to fetch neighborhoods");
                    return res.json();
                })
                .then(data => setNeighborhoodData(data))
                .catch(err => console.error("Error loading neighborhoods:", err));
        }
    }, [neighborhoodData]);

    // Fetch H3 Data
    useEffect(() => {
        if (!h3Data) {
            fetch(`${API_BASE_URL}/api/layers/h3`)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to fetch H3 data");
                    return res.json();
                })
                .then(data => setH3Data(data))
                .catch(err => console.error("Error loading H3 data:", err));
        }
    }, [h3Data]);

    // Auto-Zoom to Bbox
    useEffect(() => {
        if (zoomToBbox) {
            const [minLng, minLat, maxLng, maxLat] = zoomToBbox;
            const latitude = (minLat + maxLat) / 2;
            const longitude = (minLng + maxLng) / 2;

            // Rough zoom estimation
            const latDiff = maxLat - minLat;
            const lngDiff = maxLng - minLng;
            const maxDiff = Math.max(latDiff, lngDiff);
            let zoom = 12;
            if (maxDiff < 0.01) zoom = 15;
            else if (maxDiff < 0.05) zoom = 13;
            else if (maxDiff < 0.1) zoom = 11;
            else zoom = 10;

            setViewState({
                ...viewState,
                latitude,
                longitude,
                zoom,
                transitionDuration: 1000
            });
        }
    }, [zoomToBbox]);

    // Determine Nebula Mode
    const getMode = () => {
        switch (drawMode) {
            case 'drawPoint': return DrawPointMode;
            case 'drawLineString': return DrawLineStringMode;
            case 'drawPolygon': return DrawPolygonMode;
            case 'view':
            default: return ViewMode;
        }
    };

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
        }),
        showNeighborhoods && neighborhoodData && new GeoJsonLayer({
            id: 'neighborhoods-layer',
            data: neighborhoodData,
            pickable: true,
            stroked: true,
            filled: true,
            lineWidthMinPixels: 1,
            getLineColor: [0, 0, 0, 100],
            getFillColor: [0, 150, 255, 20],
            opacity: 0.5
        }),
        showWalkability && h3Data && new H3HexagonLayer({
            id: 'h3-layer',
            data: h3Data,
            pickable: true,
            wireframe: false,
            filled: true,
            extruded: false,
            getHexagon: d => d.h3_index,
            getFillColor: d => {
                const score = d.avg_street_score;
                // Color Scale: Red (0) -> Yellow (50) -> Green (100)
                if (score < 50) {
                    return [255, Math.floor(255 * (score / 50)), 0, 150];
                } else {
                    return [Math.floor(255 * (1 - (score - 50) / 50)), 255, 0, 150];
                }
            },
            updateTriggers: {
                getFillColor: [showWalkability]
            }
        }),
        // Drawing Layer
        new EditableGeoJsonLayer({
            id: 'drawing-layer',
            data: drawnFeatures,
            mode: getMode(),
            selectedFeatureIndexes,
            onEdit: ({ updatedData, editType }) => {
                onUpdateFeatures(updatedData);
            },
            pointRadiusMinPixels: 5,
            getFillColor: [255, 200, 0, 100],
            getLineColor: [0, 0, 0, 255],
            pickable: true,
        })
    ].filter(Boolean);

    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

    return (
        <div className="main-content" style={{ position: 'relative', width: '100%', height: '100%' }}>

            {/* Standard Map Controls */}
            <MapControls
                showNeighborhoods={showNeighborhoods}
                onToggleNeighborhoods={onToggleNeighborhoods}
                showWalkability={showWalkability}
                onToggleWalkability={onToggleWalkability}
            />

            {/* Drawing Tools Overlay - Themed */}
            <div className="drawing-tools-box">
                <div style={{ fontWeight: 'bold', fontSize: '12px', marginBottom: '5px' }}>Draw Tools</div>
                <div style={{ display: 'flex', gap: '5px' }}>
                    <button
                        onClick={() => onSetDrawMode('drawPoint')}
                        className={`tool-btn ${drawMode === 'drawPoint' ? 'active' : ''}`}
                        title="Draw Point"
                    >📍</button>
                    <button
                        onClick={() => onSetDrawMode('drawLineString')}
                        className={`tool-btn ${drawMode === 'drawLineString' ? 'active' : ''}`}
                        title="Draw Line"
                    >〰️</button>
                    <button
                        onClick={() => onSetDrawMode('drawPolygon')}
                        className={`tool-btn ${drawMode === 'drawPolygon' ? 'active' : ''}`}
                        title="Draw Polygon"
                    >⬠</button>
                    <button
                        onClick={() => onSetDrawMode('view')}
                        className={`tool-btn ${drawMode === 'view' ? 'active' : ''}`}
                        title="View Mode (Stop Drawing)"
                    >✋</button>
                </div>

                {/* Drawn Features List inside the box */}
                {drawnFeatures.features.length > 0 && (
                    <div style={{ marginTop: '10px', borderTop: '1px solid var(--border-color)', paddingTop: '5px', maxHeight: '150px', overflowY: 'auto' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginBottom: '5px' }}>Drawn Items ({drawnFeatures.features.length})</div>
                        {drawnFeatures.features.map((f, i) => (
                            <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '11px', padding: '2px 0' }}>
                                <span>{f.geometry.type} #{i + 1}</span>
                                <button
                                    onClick={() => {
                                        const newFeatures = [...drawnFeatures.features];
                                        newFeatures.splice(i, 1);
                                        onUpdateFeatures({ ...drawnFeatures, features: newFeatures });
                                    }}
                                    style={{ background: 'none', border: 'none', color: 'var(--error-color)', cursor: 'pointer', padding: '0 5px' }}
                                >✕</button>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <DeckGL
                style={{ width: '100%', height: '100%' }}
                initialViewState={undefined}
                viewState={viewState}
                onViewStateChange={({ viewState }) => setViewState(viewState)}
                controller={{ doubleClickZoom: drawMode === 'view' }}
                layers={layers}
                getTooltip={({ object }) => {
                    if (!object) return null;
                    if (object.text_description) return object.text_description;
                    if (object.properties && object.properties.name) return object.properties.name;
                    if (object.properties && object.properties.avg_walkability) return `Walkability: ${object.properties.avg_walkability.toFixed(1)}`;
                    return null;
                }}
                onClick={({ coordinate, object }) => {
                    if (drawMode === 'view' && coordinate && !object) {
                        onLocationSelect({ longitude: coordinate[0], latitude: coordinate[1] });
                    }
                }}
            >
                <Map
                    mapLib={maplibregl}
                    mapStyle={theme === 'dark' ? MAP_STYLE_DARK : MAP_STYLE_LIGHT}
                />
            </DeckGL>
        </div>
    );
}

export default MapContainer;
