package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class DeviceTest {

    @Test
    void create_shouldInitializeWithOfflineStatus() {
        Device device = new Device();
        device.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        assertEquals("AGV-001", device.getDeviceCode());
        assertEquals("搬运机器人A1", device.getDeviceName());
        assertEquals("AGV", device.getDeviceType());
        assertEquals(Device.DeviceStatus.OFFLINE, device.getStatus());
        assertNotNull(device.getCreatedAt());
        assertNotNull(device.getUpdatedAt());
    }

    @Test
    void goOnline_shouldSetStatusToOnline() {
        Device device = new Device();
        device.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        device.goOnline();
        
        assertEquals(Device.DeviceStatus.ONLINE, device.getStatus());
        assertNotNull(device.getLastHeartbeat());
    }

    @Test
    void goOffline_shouldSetStatusToOffline() {
        Device device = new Device();
        device.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        device.goOnline();
        
        device.goOffline();
        
        assertEquals(Device.DeviceStatus.OFFLINE, device.getStatus());
    }

    @Test
    void updatePosition_shouldUpdateCoordinates() {
        Device device = new Device();
        device.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        device.updatePosition(10.0, 0.25, 20.0, 1.57);
        
        assertEquals(10.0, device.getPositionX());
        assertEquals(0.25, device.getPositionY());
        assertEquals(20.0, device.getPositionZ());
        assertEquals(1.57, device.getRotationY());
    }

    @Test
    void heartbeat_shouldUpdateLastHeartbeatAndSetOnlineIfOffline() {
        Device device = new Device();
        device.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        device.heartbeat();
        
        assertNotNull(device.getLastHeartbeat());
        assertEquals(Device.DeviceStatus.ONLINE, device.getStatus());
    }

    @Test
    void equals_shouldBeBasedOnDeviceCode() {
        Device device1 = new Device();
        device1.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        Device device2 = new Device();
        device2.create("AGV-001", "搬运机器人A2", "AGV", "WS-01", "LINE-01");
        
        Device device3 = new Device();
        device3.create("AGV-002", "搬运机器人B1", "AGV", "WS-01", "LINE-01");
        
        assertEquals(device1, device2); // Same deviceCode
        assertNotEquals(device1, device3); // Different deviceCode
    }

    @Test
    void hashCode_shouldBeBasedOnDeviceCode() {
        Device device1 = new Device();
        device1.create("AGV-001", "搬运机器人A1", "AGV", "WS-01", "LINE-01");
        
        Device device2 = new Device();
        device2.create("AGV-001", "搬运机器人A2", "AGV", "WS-01", "LINE-01");
        
        assertEquals(device1.hashCode(), device2.hashCode());
    }
}