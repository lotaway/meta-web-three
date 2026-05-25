import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Environment, Text, Html } from '@react-three/drei'
import { useState, useEffect, useRef, useMemo } from 'react'
import * as THREE from 'three'

// ============= Types =============

export interface Shelf {
  id: string
  code: string
  level: number
  column: number
  capacity: number
  currentLoad: number
  status: 'empty' | 'occupied' | 'full' | 'maintenance'
}

export interface Warehouse {
  id: string
  name: string
  totalArea: number
  usedArea: number
  capacity: number
  status: 'active' | 'inactive' | 'maintenance'
}

export interface WarehouseHeatmapData {
  shelfId: string
  level: number
  column: number
  value: number // 0-100, represents activity/usage frequency
}

// ============= Shelf Component =============

function ShelfModel({ 
  shelf, 
  isSelected, 
  onClick,
  heatmapValue 
}: { 
  shelf: Shelf
  isSelected: boolean
  onClick: () => void
  heatmapValue?: number
}) {
  const groupRef = useRef<THREE.Group>(null)
  
  // Calculate color based on load and heatmap
  const getColor = () => {
    if (shelf.status === 'maintenance') return '#fbbf24'
    if (shelf.status === 'empty') return '#334155'
    if (shelf.status === 'full') return '#ef4444'
    
    // Heatmap coloring: blue (cold) -> green -> yellow -> red (hot)
    if (heatmapValue !== undefined) {
      if (heatmapValue < 25) return '#3b82f6'
      if (heatmapValue < 50) return '#22c55e'
      if (heatmapValue < 75) return '#eab308'
      return '#ef4444'
    }
    
    // Load-based coloring
    const loadRatio = shelf.currentLoad / shelf.capacity
    if (loadRatio < 0.3) return '#22c55e'
    if (loadRatio < 0.7) return '#eab308'
    return '#ef4444'
  }

  const getStatusIndicator = () => {
    switch (shelf.status) {
      case 'full': return '#ef4444'
      case 'occupied': return '#22c55e'
      case 'empty': return '#64748b'
      case 'maintenance': return '#fbbf24'
      default: return '#3b82f6'
    }
  }

  // Shelf frame dimensions
  const shelfWidth = 1.8
  const shelfHeight = 2.5
  const shelfDepth = 0.6
  const levels = 4

  return (
    <group 
      ref={groupRef}
      position={[shelf.column * 2.5, 0, shelf.level * 3]}
      onClick={(e) => { e.stopPropagation(); onClick() }}
    >
      {/* Shelf frame - vertical posts */}
      {/* Left post */}
      <mesh position={[-shelfWidth/2, shelfHeight/2, 0]}>
        <boxGeometry args={[0.1, shelfHeight, 0.1]} />
        <meshStandardMaterial color="#475569" metalness={0.6} roughness={0.4} />
      </mesh>
      {/* Right post */}
      <mesh position={[shelfWidth/2, shelfHeight/2, 0]}>
        <boxGeometry args={[0.1, shelfHeight, 0.1]} />
        <meshStandardMaterial color="#475569" metalness={0.6} roughness={0.4} />
      </mesh>
      
      {/* Shelves (levels) */}
      {Array.from({ length: levels }).map((_, levelIdx) => (
        <group key={levelIdx}>
          {/* Shelf surface */}
          <mesh position={[0, (levelIdx + 0.5) * (shelfHeight / levels), 0]}>
            <boxGeometry args={[shelfWidth, 0.05, shelfDepth]} />
            <meshStandardMaterial 
              color={getColor()} 
              metalness={0.3} 
              roughness={0.6}
              emissive={isSelected ? getColor() : '#000000'}
              emissiveIntensity={isSelected ? 0.2 : 0}
            />
          </mesh>
          
          {/* Load indicator bars */}
          {shelf.status !== 'empty' && (
            <mesh 
              position={[0, (levelIdx + 0.5) * (shelfHeight / levels) + 0.1, 0]}
            >
              <boxGeometry args={[shelfWidth * (shelf.currentLoad / shelf.capacity), 0.15, shelfDepth * 0.8]} />
              <meshStandardMaterial 
                color={getStatusIndicator()} 
                emissive={getStatusIndicator()}
                emissiveIntensity={0.3}
              />
            </mesh>
          )}
        </group>
      ))}
      
      {/* Status indicator light */}
      <mesh position={[shelfWidth/2 + 0.1, shelfHeight + 0.1, 0]}>
        <sphereGeometry args={[0.08, 16, 16]} />
        <meshStandardMaterial 
          color={getStatusIndicator()} 
          emissive={getStatusIndicator()}
          emissiveIntensity={shelf.status === 'active' || shelf.status === 'occupied' ? 1 : 0.3}
        />
      </mesh>
      
      {/* Shelf code label */}
      <Text
        position={[0, shelfHeight + 0.3, shelfDepth/2 + 0.1]}
        fontSize={0.2}
        color="#94a3b8"
        anchorX="center"
        anchorY="bottom"
      >
        {shelf.code}
      </Text>
    </group>
  )
}

// ============= Warehouse Floor =============

function WarehouseFloor({ area }: { area: number }) {
  const size = Math.sqrt(area)
  
  return (
    <group>
      {/* Main floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]} receiveShadow>
        <planeGeometry args={[size, size]} />
        <meshStandardMaterial color="#1e293b" />
      </mesh>
      
      {/* Floor grid lines */}
      <gridHelper args={[size, Math.floor(size), '#334155', '#1e293b']} position={[0, 0.02, 0]} />
    </group>
  )
}

// ============= Warehouse Walls =============

function WarehouseWalls({ area, height = 5 }: { area: number, height?: number }) {
  const size = Math.sqrt(area)
  const wallColor = "#475569"
  
  return (
    <group>
      {/* Back wall */}
      <mesh position={[0, height/2, -size/2]}>
        <boxGeometry args={[size, height, 0.1]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.3} />
      </mesh>
      
      {/* Left wall */}
      <mesh position={[-size/2, height/2, 0]}>
        <boxGeometry args={[0.1, height, size]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.3} />
      </mesh>
      
      {/* Right wall */}
      <mesh position={[size/2, height/2, 0]}>
        <boxGeometry args={[0.1, height, size]} />
        <meshStandardMaterial color={wallColor} transparent opacity={0.3} />
      </mesh>
    </group>
  )
}

// ============= Heatmap Overlay =============

function HeatmapOverlay({ 
  data, 
  warehouseArea 
}: { 
  data: WarehouseHeatmapData[]
  warehouseArea: number
}) {
  const size = Math.sqrt(warehouseArea)
  
  // Create heatmap texture from data
  const heatmapTexture = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    
    // Initialize with transparent
    ctx.fillStyle = 'rgba(0, 0, 0, 0)'
    ctx.fillRect(0, 0, 64, 64)
    
    // Draw heat points
    data.forEach(point => {
      const x = (point.column / 20) * 64
      const y = (point.level / 10) * 64
      const intensity = point.value / 100
      
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, 8)
      const color = intensity < 0.25 
        ? `rgba(59, 130, 246, ${intensity * 4})`
        : intensity < 0.5 
          ? `rgba(34, 197, 94, ${(intensity - 0.25) * 4})`
          : intensity < 0.75 
            ? `rgba(234, 179, 8, ${(intensity - 0.5) * 4})`
            : `rgba(239, 68, 68, ${(intensity - 0.75) * 4})`
      
      gradient.addColorStop(0, color)
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
      ctx.fillStyle = gradient
      ctx.fillRect(x - 8, y - 8, 16, 16)
    })
    
    const texture = new THREE.CanvasTexture(canvas)
    texture.needsUpdate = true
    return texture
  }, [data])
  
  if (!heatmapTexture) return null
  
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.05, 0]}>
      <planeGeometry args={[size, size]} />
      <meshBasicMaterial 
        map={heatmapTexture} 
        transparent 
        opacity={0.5} 
        depthWrite={false}
      />
    </mesh>
  )
}

// ============= Main Component =============

export interface Warehouse3DViewProps {
  warehouse: Warehouse
  shelves: Shelf[]
  heatmapData?: WarehouseHeatmapData[]
  onShelfClick?: (shelf: Shelf) => void
  selectedShelfId?: string
  showHeatmap?: boolean
  showGrid?: boolean
}

export function Warehouse3DView({ 
  warehouse, 
  shelves, 
  heatmapData = [],
  onShelfClick,
  selectedShelfId,
  showHeatmap = false,
  showGrid = true
}: Warehouse3DViewProps) {
  const [hoveredShelf, setHoveredShelf] = useState<string | null>(null)
  
  // Calculate camera position based on warehouse size
  const warehouseSize = Math.sqrt(warehouse.totalArea)
  const cameraDistance = warehouseSize * 1.5
  
  // Get heatmap value for a shelf
  const getHeatmapValue = (shelfId: string): number | undefined => {
    if (!showHeatmap) return undefined
    const data = heatmapData.find(d => d.shelfId === shelfId)
    return data?.value
  }

  return (
    <Canvas
      shadows
      camera={{ position: [cameraDistance, cameraDistance * 0.8, cameraDistance], fov: 50 }}
      style={{ background: '#0f172a' }}
    >
      <OrbitControls 
        makeDefault 
        maxPolarAngle={Math.PI / 2.1}
        minDistance={5}
        maxDistance={cameraDistance * 2}
      />
      
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight 
        position={[10, 20, 10]} 
        intensity={1} 
        castShadow 
        shadow-mapSize={[2048, 2048]}
      />
      <pointLight position={[-10, 10, -10]} intensity={0.3} color="#60a5fa" />
      <pointLight position={[10, 10, 10]} intensity={0.3} color="#fbbf24" />
      
      {/* Environment */}
      <Environment preset="warehouse" />
      
      {showGrid && (
        <Grid 
          args={[warehouseSize * 1.5, warehouseSize * 1.5]} 
          cellSize={1} 
          cellThickness={0.5} 
          cellColor="#334155" 
          sectionSize={5}
          sectionThickness={1}
          sectionColor="#475569"
          fadeDistance={50}
          infiniteGrid
        />
      )}
      
      {/* Warehouse structure */}
      <WarehouseFloor area={warehouse.totalArea} />
      <WarehouseWalls area={warehouse.totalArea} height={5} />
      
      {/* Heatmap overlay */}
      {showHeatmap && heatmapData.length > 0 && (
        <HeatmapOverlay data={heatmapData} warehouseArea={warehouse.totalArea} />
      )}
      
      {/* Shelves */}
      {shelves.map((shelf) => (
        <ShelfModel
          key={shelf.id}
          shelf={shelf}
          isSelected={shelf.id === selectedShelfId}
          onClick={() => onShelfClick?.(shelf)}
          heatmapValue={getHeatmapValue(shelf.id)}
        />
      ))}
      
      {/* Warehouse info panel */}
      <Html position={[warehouseSize/2 - 2, 3, -warehouseSize/2 + 1]} center>
        <div style={{ 
          background: 'rgba(15, 23, 42, 0.9)', 
          padding: '12px 16px', 
          borderRadius: '8px',
          border: '1px solid #334155',
          color: '#f1f5f9',
          fontSize: '12px',
          fontFamily: 'monospace',
          minWidth: '180px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '8px', color: '#38bdf8' }}>
            {warehouse.name}
          </div>
          <div style={{ display: 'grid', gap: '4px' }}>
            <div>总面积: <span style={{ color: '#94a3b8' }}>{warehouse.totalArea} m²</span></div>
            <div>使用面积: <span style={{ color: '#22c55e' }}>{warehouse.usedArea} m²</span></div>
            <div>利用率: <span style={{ color: warehouse.usedArea/warehouse.totalArea > 0.8 ? '#ef4444' : '#22c55e' }}>
              {((warehouse.usedArea / warehouse.totalArea) * 100).toFixed(1)}%
            </span></div>
            <div>容量: <span style={{ color: '#60a5fa' }}>{warehouse.capacity}</span></div>
            <div>状态: <span style={{ 
              color: warehouse.status === 'active' ? '#22c55e' : 
                     warehouse.status === 'maintenance' ? '#fbbf24' : '#94a3b8' 
            }}>
              {warehouse.status.toUpperCase()}
            </span></div>
          </div>
        </div>
      </Html>
    </Canvas>
  )
}

export type { Shelf, Warehouse, WarehouseHeatmapData, Warehouse3DViewProps }