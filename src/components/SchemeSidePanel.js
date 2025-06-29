import React from 'react';
import './SchemeSidePanel.css';

const SchemeSidePanel = ({ scheme, isOpen, onClose }) => {
  if (!scheme) return null;

  // Helper function to format parameter names for display
  const formatParameterName = (name) => {
    if (!name) return '';
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  // Ensure we have safe objects to work with
  const parameters = scheme.parameters || {};
  const evaluations = scheme.evaluations || {};

  return (
    <div className={`scheme-side-panel ${isOpen ? 'open' : ''}`}>
      <div className="side-panel-header">
        <h2>Scheme {scheme.id || ''} Details</h2>
        <button className="close-panel-button" onClick={onClose}>×</button>
      </div>
      
      <div className="side-panel-content">
        {/* Parameters Section */}
        <div className="details-section">
          <h3>Parameters</h3>
          <div className="details-list">
            {Object.keys(parameters).length > 0 ? (
              Object.entries(parameters).map(([key, value]) => (
                <div className="detail-item" key={key}>
                  <div className="detail-name">{formatParameterName(key)}</div>
                  <div className="detail-value">{String(value)}</div>
                </div>
              ))
            ) : (
              <div className="detail-item">
                <div className="detail-name">No parameters available</div>
                <div className="detail-value">-</div>
              </div>
            )}
          </div>
        </div>
        
        {/* Evaluations Section */}
        <div className="details-section">
          <h3>Evaluations</h3>
          <div className="details-list">
            {Object.keys(evaluations).length > 0 ? (
              Object.entries(evaluations).map(([key, value]) => (
                <div className="detail-item" key={key}>
                  <div className="detail-name">{formatParameterName(key)}</div>
                  <div className="detail-value">{String(value)}</div>
                </div>
              ))
            ) : (
              <div className="detail-item">
                <div className="detail-name">No evaluations available</div>
                <div className="detail-value">-</div>
              </div>
            )}
          </div>
        </div>
        
        {/* Visualization Properties Section */}
        <div className="details-section">
          <h3>Visualization Properties</h3>
          <div className="details-list">
            <div className="detail-item">
              <div className="detail-name">Width</div>
              <div className="detail-value">{scheme.width || 'Default'}</div>
            </div>
            <div className="detail-item">
              <div className="detail-name">Height</div>
              <div className="detail-value">{scheme.height || 'Default'}</div>
            </div>
            <div className="detail-item">
              <div className="detail-name">Depth</div>
              <div className="detail-value">{scheme.depth || 'Default'}</div>
            </div>
            <div className="detail-item">
              <div className="detail-name">Color</div>
              <div className="detail-value">
                <span className="color-preview" style={{ backgroundColor: scheme.color || '#cccccc' }}></span>
                {scheme.color || 'Default'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SchemeSidePanel; 