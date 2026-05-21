import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid, Environment } from '@react-three/drei'
import { useState, useEffect } from 'react'
import * as THREE from 'three'

interface Device {
  id: string
  code: string
  name: string
  type: string
  status: 'online' | 'offline' | 'running' | 'idle' | 'warning' | 'error'
  position: [number, number, number]
  rotation: number
}

interface FactorySceneProps {
  devices: Device[]
  onDeviceClick?: (device: Device) => void
  selectedDeviceId?: string
}

function DeviceModel({ device, isSelected, onClick }: { 
  device: Device
  isSelected: boolean
  onClick: () => void
}) {
  const getColor = () => {
    switch (device.status) {
      case 'running': return '#4ade80'
      case 'idle': return '#94a3b8'
      case 'warning': return '#fbbf24'
      case 'error': return '#ef4444'
      case 'offline': return '#6b7280'
      default: return '#3b82f6'
    }
  }

  const getGeometry = () => {
    switch (device.type) {
      case 'AGV': return <boxGeometry args={[1, 0.5, 1.5]} />
      case 'ROBOT': return <cylinderGeometry args={[0.3, 0.3, 1.5, 16]} />
      case 'PLC': return <boxGeometry args={[0.8, 0.6, 0.4]} />
      case 'CONVEYOR': return <boxGeometry args={[2, 0.3, 0.5]} />
      default: return <boxGeometry args={[1, 1, 1]} />
    }
  }

  return (
    <group 
      position={device.position} 
      rotation={[0, device.rotation, 0]}
      onClick={(e) => { e.stopPropagation(); onClick() }}
    >
      <mesh>
        {getGeometry()}
        <meshStandardMaterial 
          color={getColor()} 
          emissive={isSelected ? '#ffffff' : getColor()}
          emissiveIntensity={isSelected ? 0.3 : 0.1}
        />
      </mesh>
      {/* Status indicator */}
      <mesh position={[0, 1, 0]}>
        <sphereGeometry args={[0.1, 16, 16]} />
        <meshStandardMaterial color={getColor()} />
      </mesh>
    </group>
  )
}

export function FactoryScene({ devices, onDeviceClick, selectedDeviceId }: FactorySceneProps) {
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([10, 10, 10])

  return (
    <div style={{ width: '100%', height: '100%', background: '#1a1a2e' }}>
      <Canvas
        camera={{ position: cameraPosition, fov: 50 }}
        onCreated={({ camera }) => {
          camera.lookAt(0, 0, 0)
        }}
      >
        <ambientLight intensity={0.5} />
        <directionalLight position={[10, 10, 5]} intensity={1} />
        <pointLight position={[-10, 10, -10]} intensity={0.5} />
        
        {/* Floor grid */}
        <Grid 
          args={[50, 50]} 
          cellSize={1} 
          cellThickness={0.5} 
          cellColor="#4a5568" 
          sectionSize={5} 
          sectionThickness={1}
          sectionColor="#718096"
          fadeDistance={30}
          fadeStrength={1}
        />
        
        {/* Factory floor */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.01, 0]}>
          <planeGeometry args={[50, 50]} />
          <meshStandardMaterial color="#2d3748" />
        </mesh>

        {/* Devices */}
        {devices.map((device) => (
          <DeviceModel
            key={device.id}
            device={device}
            isSelected={device.id === selectedDeviceId}
            onClick={() => onDeviceClick?.(device)}
          />
        ))}

        {/* Camera controls */}
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={50}
          maxPolarAngle={Math.PI / 2.1}
        />
        
        {/* Environment */}
        <Environment preset="city" />
      </Canvas>
    </div>
  )
}

export type { Device, FactorySceneProps }