import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import CuboidModel from './CuboidModel';
import SchemeSidePanel from './SchemeSidePanel';
import './SchemeGrid.css';

// Component for a single scheme with frame and title
const SchemeFrame = ({ scheme, index, onSchemeClick }) => {
  // Debug log for individual scheme
  console.log(`Scheme ${index} data:`, scheme);
  
  // Ensure scheme has default values to prevent undefined errors
  const safeScheme = {
    id: scheme.id || index + 1,
    width: scheme.width || 1,
    height: scheme.height || 1,
    depth: scheme.depth || 1,
    color: scheme.color || "#4287f5",
  };
  
  // Debug log for processed safeScheme
  console.log(`Scheme ${index} processed:`, safeScheme);

  // Scale down dimensions for better visualization
  const scaleFactor = 10;
  const scaledWidth = safeScheme.width / scaleFactor;
  const scaledHeight = safeScheme.height / scaleFactor;
  const scaledDepth = safeScheme.depth / scaleFactor;
  
  console.log(`Scheme ${index} scaled dimensions:`, { 
    width: scaledWidth, 
    height: scaledHeight, 
    depth: scaledDepth 
  });

  return (
    <div 
      className="scheme-frame" 
      onClick={() => onSchemeClick(safeScheme, index)}
    >
      <div className="scheme-title">Scheme {safeScheme.id}</div>
      <div className="scheme-canvas-container">
        <Canvas camera={{ position: [0, 2, 5], fov: 50 }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[5, 5, 5]} intensity={1} />
          <directionalLight position={[-5, -5, -5]} intensity={0.3} />
          
          <CuboidModel 
            width={scaledWidth}
            height={scaledHeight}
            depth={scaledDepth}
            position={[0, 0, 0]}
            color={safeScheme.color}
          />
          
          {/* Ground plane */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]} receiveShadow>
            <planeGeometry args={[10, 10]} />
            <shadowMaterial opacity={0.2} />
          </mesh>
          
          <Environment preset="city" />
          <OrbitControls 
            enablePan={false} 
            minDistance={3}
            maxDistance={8}
          />
        </Canvas>
      </div>
    </div>
  );
};

// Empty state component when no schemes are available
const EmptyState = () => {
  return (
    <div className="empty-state">
      <div className="empty-state-content">
        <h3>No Schemes Available</h3>
        <p>Enter a query to generate building schemes.</p>
        <p>Try something like: "Generate a building scheme with 5 floors"</p>
      </div>
    </div>
  );
};

// Main component for the grid of schemes
const SchemeGrid = ({ schemes = [] }) => {
  const [selectedScheme, setSelectedScheme] = useState(null);
  const [isSidePanelOpen, setIsSidePanelOpen] = useState(false);

  // Debug log to see the schemes data structure
  console.log("Schemes received in SchemeGrid:", schemes);
  
  const handleSchemeClick = (scheme, index) => {
    // Make sure the scheme has an ID
    const safeScheme = {
      ...scheme,
      id: scheme.id || index + 1
    };
    setSelectedScheme(safeScheme);
    setIsSidePanelOpen(true);
  };

  const handleCloseSidePanel = () => {
    setIsSidePanelOpen(false);
  };

  // If no schemes, show empty state
  if (!schemes || !Array.isArray(schemes) || schemes.length === 0) {
    return <EmptyState />;
  }

  // Calculate grid layout based on number of schemes
  let gridClassName = "scheme-grid";
  if (schemes.length === 1) {
    gridClassName += " single-scheme";
  } else if (schemes.length === 2) {
    gridClassName += " two-schemes";
  } else if (schemes.length <= 4) {
    gridClassName += " four-schemes";
  }

  return (
    <div className="scheme-grid-container">
      <div className={gridClassName}>
        {schemes.map((scheme, index) => (
          <SchemeFrame 
            key={`scheme-${index}`} 
            scheme={scheme} 
            index={index}
            onSchemeClick={handleSchemeClick}
          />
        ))}
      </div>
      
      <SchemeSidePanel 
        scheme={selectedScheme}
        isOpen={isSidePanelOpen}
        onClose={handleCloseSidePanel}
      />
    </div>
  );
};

export default SchemeGrid; 