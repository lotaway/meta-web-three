import { useState, useEffect } from 'react'
import styled from 'styled-components'
import { FactoryScene, Device, Alert, DeviceStatus, AlertPanel, DeviceChart, StatsCard } from '../components/digital-twin'
import { useDigitalTwinWebSocket } from '../services/websocket'

const PageContainer = styled.div`
  width: 100vw;
  height: 100vh;
  display: flex;
  background: #0f172a;
  color: #e2e8f0;
`

const LeftPanel = styled.div`
  width: 320px;
  background: #1e293b;
  border-right: 1px solid #334155;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`

const MainArea = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`

const SceneContainer = styled.div`
  flex: 1;
  position: relative;
`

const RightPanel = styled.div`
  width: 360px;
  background: #1e293b;
  border-left: 1px solid #334155;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`

const TabBar = styled.div`
  display: flex;
  background: #0f172a;
  border-bottom: 1px solid #334155;
`

const Tab = styled.button<{ $active: boolean }>`
  flex: 1;
  padding: 12px;
  background: ${({ $active }) => $active ? '#1e293b' : '#0f172a'};
  border: none;
  color: ${({ $active }) => $active ? '#3b82f6' : '#94a3b8'};
  cursor: pointer;
  font-weight: ${({ $active }) => $active ? 'bold' : 'normal'};
  border-bottom: 2px solid ${({ $active }) => $active ? '#3b82f6' : 'transparent'};
  
  &:hover {
    background: #1e293b;
  }
`

const PanelContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 12px;
`

const StatsRow = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
  margin-bottom: 16px;
`

const Header = styled.div`
  padding: 16px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
  display: flex;
  justify-content: space-between;
  align-items: center;
`

const Title = styled.h2`
  margin: 0;
  font-size: 18px;
  color: #3b82f6;
`

const ConnectionStatus = styled.span<{ $connected: boolean }>`
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  background: ${({ $connected }) => $connected ? '#10b981' : '#ef4444'};
  color: white;
`

const CloseButton = styled.button`
  padding: 8px 16px;
  background: #ef4444;
  border: none;
  border-radius: 4px;
  color: white;
  cursor: pointer;
  
  &:hover {
    background: #dc2626;
  }
`

interface DigitalTwinPageProps {
  onClose?: () => void
}

// Mock data for demo
const mockDevices: Device[] = [
  { id: '1', code: 'AGV-001', name: '搬运机器人A1', type: 'AGV', status: 'running', position: [2, 0.25, 3], rotation: 0 },
  { id: '2', code: 'AGV-002', name: '搬运机器人A2', type: 'AGV', status: 'running', position: [-3, 0.25, 5], rotation: Math.PI / 4 },
  { id: '3', code: 'ROBOT-001', name: '机械臂R1', type: 'ROBOT', status: 'running', position: [5, 0.75, -2], rotation: Math.PI },
  { id: '4', code: 'PLC-001', name: 'PLC控制器C1', type: 'PLC', status: 'online', position: [-5, 0.3, -3], rotation: 0 },
  { id: '5', code: 'CONVEYOR-001', name: '传送带S1', type: 'CONVEYOR', status: 'running', position: [0, 0.15, 0], rotation: 0 },
  { id: '6', code: 'AGV-003', name: '搬运机器人A3', type: 'AGV', status: 'idle', position: [-2, 0.25, -4], rotation: Math.PI / 2 },
  { id: '7', code: 'ROBOT-002', name: '机械臂R2', type: 'ROBOT', status: 'warning', position: [7, 0.75, 2], rotation: -Math.PI / 4 },
  { id: '8', code: 'PLC-002', name: 'PLC控制器C2', type: 'PLC', status: 'error', position: [-7, 0.3, 4], rotation: 0 },
]

const mockAlerts: Alert[] = [
  { id: '1', code: 'ALT-001', deviceCode: 'PLC-002', deviceName: 'PLC控制器C2', level: 'critical', type: 'DEVICE_ERROR', title: '设备故障', description: 'PLC控制器C2通信异常', status: 'triggered', occurredAt: new Date().toISOString() },
  { id: '2', code: 'ALT-002', deviceCode: 'ROBOT-002', deviceName: '机械臂R2', level: 'warning', type: 'TEMPERATURE_HIGH', title: '温度告警', description: '机械臂R2温度过高', status: 'acknowledged', occurredAt: new Date(Date.now() - 300000).toISOString() },
  { id: '3', code: 'ALT-003', deviceCode: 'AGV-001', deviceName: '搬运机器人A1', level: 'info', type: 'MAINTENANCE_DUE', title: '维护提醒', description: '搬运机器人A1即将到期维护', status: 'triggered', occurredAt: new Date(Date.now() - 600000).toISOString() },
]

const mockChartData = Array.from({ length: 20 }, (_, i) => ({
  timestamp: Date.now() - (20 - i) * 60000,
  value: 80 + Math.random() * 20
}))

export default function DigitalTwinPage({ onClose }: DigitalTwinPageProps) {
  const [activeTab, setActiveTab] = useState<'devices' | 'alerts' | 'charts'>('devices')
  const [devices, setDevices] = useState<Device[]>(mockDevices)
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts)
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null)
  
  // WebSocket connection (using mock URL for demo)
  const { isConnected } = useDigitalTwinWebSocket('ws://localhost:8080/ws/digital-twin')

  const handleDeviceClick = (device: Device) => {
    setSelectedDevice(device)
  }

  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(a => 
      a.id === alertId ? { ...a, status: 'acknowledged' as const } : a
    ))
  }

  const handleResolveAlert = (alertId: string) => {
    setAlerts(prev => prev.map(a => 
      a.id === alertId ? { ...a, status: 'resolved' as const } : a
    ))
  }

  return (
    <PageContainer>
      {/* Left Panel - Device List */}
      <LeftPanel>
        <Header>
          <Title>设备列表</Title>
          <ConnectionStatus $connected={isConnected}>
            {isConnected ? '已连接' : '未连接'}
          </ConnectionStatus>
        </Header>
        <PanelContent>
          <DeviceStatus 
            devices={devices} 
            onDeviceSelect={handleDeviceClick}
          />
        </PanelContent>
      </LeftPanel>

      {/* Main Area - 3D Scene */}
      <MainArea>
        <Header>
          <Title>🏭 数字孪生工厂</Title>
          <CloseButton onClick={onClose}>退出</CloseButton>
        </Header>
        <SceneContainer>
          <FactoryScene 
            devices={devices}
            onDeviceClick={handleDeviceClick}
            selectedDeviceId={selectedDevice?.id}
          />
        </SceneContainer>
      </MainArea>

      {/* Right Panel - Details */}
      <RightPanel>
        <TabBar>
          <Tab $active={activeTab === 'devices'} onClick={() => setActiveTab('devices')}>
            设备详情
          </Tab>
          <Tab $active={activeTab === 'alerts'} onClick={() => setActiveTab('alerts')}>
            告警 {alerts.filter(a => a.status === 'triggered').length}
          </Tab>
          <Tab $active={activeTab === 'charts'} onClick={() => setActiveTab('charts')}>
            图表
          </Tab>
        </TabBar>

        <PanelContent>
          {activeTab === 'devices' && (
            <div>
              {selectedDevice ? (
                <div>
                  <h3 style={{ margin: '0 0 16px 0', color: '#3b82f6' }}>{selectedDevice.name}</h3>
                  <StatsRow>
                    <StatsCard 
                      title="状态" 
                      value={selectedDevice.status === 'running' ? '运行中' : selectedDevice.status === 'idle' ? '空闲' : '故障'} 
                      color={selectedDevice.status === 'running' ? '#10b981' : '#f59e0b'}
                    />
                    <StatsCard title="类型" value={selectedDevice.type} color="#3b82f6" />
                  </StatsRow>
                  <StatsRow>
                    <StatsCard title="X" value={selectedDevice.position[0].toFixed(1)} unit="m" color="#8b5cf6" />
                    <StatsCard title="Z" value={selectedDevice.position[2].toFixed(1)} unit="m" color="#8b5cf6" />
                  </StatsRow>
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ fontSize: '12px', color: '#94a3b8', marginBottom: '8px' }}>实时产量</div>
                    <DeviceChart 
                      title="设备产量趋势" 
                      data={mockChartData}
                      unit="件"
                      color="#10b981"
                    />
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                  点击设备查看详情
                </div>
              )}
            </div>
          )}

          {activeTab === 'alerts' && (
            <AlertPanel
              alerts={alerts}
              onAcknowledge={handleAcknowledgeAlert}
              onResolve={handleResolveAlert}
            />
          )}

          {activeTab === 'charts' && (
            <div>
              <StatsRow>
                <StatsCard title="在线设备" value={devices.filter(d => d.status !== 'offline').length} color="#10b981" />
                <StatsCard title="运行中" value={devices.filter(d => d.status === 'running').length} color="#3b82f6" />
              </StatsRow>
              <StatsRow>
                <StatsCard title="告警" value={alerts.filter(a => a.status === 'triggered').length} color="#ef4444" />
                <StatsCard title="效率" value="87.5" unit="%" color="#f59e0b" change={2.3} />
              </StatsRow>
              <DeviceChart 
                title="今日产量" 
                data={mockChartData}
                unit="件"
                color="#3b82f6"
              />
              <div style={{ height: '16px' }} />
              <DeviceChart 
                title="设备利用率" 
                data={mockChartData.map(d => ({ ...d, value: 60 + Math.random() * 30 }))}
                unit="%"
                color="#8b5cf6"
              />
            </div>
          )}
        </PanelContent>
      </RightPanel>
    </PageContainer>
  )
}