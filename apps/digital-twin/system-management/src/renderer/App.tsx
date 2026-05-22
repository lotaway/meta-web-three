import { useState } from 'react'
import Sidebar from './components/Sidebar'
import { ShapeType } from '../types/Editor'
import Settings from './pages/Settings'
import Editor from './components/Editor'
import DigitalTwinPage from './pages/DigitalTwinPage'
import { AppContainer, SettingsToggle, OverlayLayer, OverlayBody } from './components/AppLayout'
import styled from 'styled-components'

const DigitalTwinToggle = styled.button<{ $active: boolean }>`
    position: absolute;
    top: 10px;
    right: 140px;
    z-index: 2000;
    padding: 8px 16px;
    border-radius: 4px;
    border: none;
    background-color: ${({ $active }: { $active: boolean }) => ($active ? '#7c3aed' : '#3b82f6')};
    color: white;
    cursor: pointer;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    
    &:hover {
        background-color: ${({ $active }: { $active: boolean }) => ($active ? '#6d28d9' : '#2563eb')};
    }
`

const ToggleButton = ({ show, onClick }: { show: boolean; onClick: () => void }) => (
    <SettingsToggle $active={show} onClick={onClick}>
        {show ? 'Close Settings' : 'Settings'}
    </SettingsToggle>
)

const SettingsOverlay = ({ show }: { show: boolean }) => {
    if (!show) return null
    return (
        <OverlayLayer>
            <OverlayBody>
                <Settings />
            </OverlayBody>
        </OverlayLayer>
    )
}

export default function App() {
    const [dragType, setDragType] = useState<ShapeType | null>(null)
    const [showSettings, setShowSettings] = useState(false)
    const [showDigitalTwin, setShowDigitalTwin] = useState(false)

    if (showDigitalTwin) {
        return <DigitalTwinPage onClose={() => setShowDigitalTwin(false)} />
    }

    return (
        <AppContainer>
            <DigitalTwinToggle $active={showDigitalTwin} onClick={() => setShowDigitalTwin(true)}>
                🏭 数字孪生
            </DigitalTwinToggle>
            <ToggleButton show={showSettings} onClick={() => setShowSettings(!showSettings)} />
            <SettingsOverlay show={showSettings} />
            <Sidebar />
            <Editor dragType={dragType} setDragType={setDragType} />
        </AppContainer>
    )
}
