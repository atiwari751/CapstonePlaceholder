import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';

const CuboidModel = ({ width = 1, height = 1, depth = 1, position = [0, 0, 0], color = "#4287f5" }) => {
  // Debug log for CuboidModel props
  console.log("CuboidModel props:", { width, height, depth, position, color });
  
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const [clicked, setClicked] = useState(false);
  
  // Simple animation - gentle floating effect
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      
      if (clicked) {
        // Rotate when clicked
        meshRef.current.rotation.y += 0.02;
      }
    }
  });
  
  return (
    <mesh
      ref={meshRef}
      position={position}
      scale={clicked ? 1.1 : 1}
      onClick={() => setClicked(!clicked)}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      castShadow
    >
      <boxGeometry args={[width, height, depth]} />
      <meshStandardMaterial 
        color={hovered ? '#ffffff' : color} 
        metalness={0.5}
        roughness={0.2}
      />
    </mesh>
  );
};

export default CuboidModel; 