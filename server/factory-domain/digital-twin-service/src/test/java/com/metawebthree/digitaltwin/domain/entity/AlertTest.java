package com.metawebthree.digitaltwin.domain.entity;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;


class AlertTest {

    @Test
    void create_shouldInitializeWithTriggeredStatus() {
        Alert alert = new Alert();
        alert.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "设备故障", "PLC控制器C2通信异常");
        
        assertEquals("ALT-001", alert.getAlertCode());
        assertEquals("AGV-001", alert.getDeviceCode());
        assertEquals("WS-01", alert.getWorkshopId());
        assertEquals(Alert.AlertLevel.CRITICAL, alert.getLevel());
        assertEquals(Alert.AlertType.DEVICE_ERROR, alert.getType());
        assertEquals("设备故障", alert.getTitle());
        assertEquals("PLC控制器C2通信异常", alert.getDescription());
        assertEquals(Alert.AlertStatus.TRIGGERED, alert.getStatus());
        assertNotNull(alert.getOccurredAt());
    }

    @Test
    void acknowledge_shouldChangeStatusToAcknowledged() {
        Alert alert = new Alert();
        alert.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "设备故障", "PLC控制器C2通信异常");
        
        alert.acknowledge("admin");
        
        assertEquals(Alert.AlertStatus.ACKNOWLEDGED, alert.getStatus());
        assertEquals("admin", alert.getAcknowledgedBy());
        assertNotNull(alert.getAcknowledgedAt());
    }

    @Test
    void resolve_shouldChangeStatusToResolved() {
        Alert alert = new Alert();
        alert.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "设备故障", "PLC控制器C2通信异常");
        
        alert.resolve("admin", "已更换PLC控制器");
        
        assertEquals(Alert.AlertStatus.RESOLVED, alert.getStatus());
        assertEquals("admin", alert.getResolvedBy());
        assertEquals("已更换PLC控制器", alert.getSolution());
        assertNotNull(alert.getResolvedAt());
    }

    @Test
    void close_shouldChangeStatusToClosed() {
        Alert alert = new Alert();
        alert.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "设备故障", "PLC控制器C2通信异常");
        alert.resolve("admin", "已解决");
        
        alert.close();
        
        assertEquals(Alert.AlertStatus.CLOSED, alert.getStatus());
    }

    @Test
    void escalate_shouldIncreaseAlertLevel() {
        Alert alert = new Alert();
        alert.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.INFO, Alert.AlertType.MAINTENANCE_DUE,
            "维护提醒", "设备需要维护");
        
        alert.escalate();
        
        assertEquals(Alert.AlertLevel.WARNING, alert.getLevel());
        
        alert.escalate();
        assertEquals(Alert.AlertLevel.ERROR, alert.getLevel());
        
        alert.escalate();
        assertEquals(Alert.AlertLevel.CRITICAL, alert.getLevel());
        
        // Should not go beyond CRITICAL
        alert.escalate();
        assertEquals(Alert.AlertLevel.CRITICAL, alert.getLevel());
    }

    @Test
    void equals_shouldBeBasedOnAlertCode() {
        Alert alert1 = new Alert();
        alert1.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "标题1", "描述1");
        
        Alert alert2 = new Alert();
        alert2.create("ALT-001", "AGV-002", "WS-02", 
            Alert.AlertLevel.INFO, Alert.AlertType.MAINTENANCE_DUE,
            "标题2", "描述2");
        
        Alert alert3 = new Alert();
        alert3.create("ALT-002", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "标题1", "描述1");
        
        assertEquals(alert1, alert2); // Same alertCode
        assertNotEquals(alert1, alert3); // Different alertCode
    }

    @Test
    void hashCode_shouldBeBasedOnAlertCode() {
        Alert alert1 = new Alert();
        alert1.create("ALT-001", "AGV-001", "WS-01", 
            Alert.AlertLevel.CRITICAL, Alert.AlertType.DEVICE_ERROR,
            "标题1", "描述1");
        
        Alert alert2 = new Alert();
        alert2.create("ALT-001", "AGV-002", "WS-02", 
            Alert.AlertLevel.INFO, Alert.AlertType.MAINTENANCE_DUE,
            "标题2", "描述2");
        
        assertEquals(alert1.hashCode(), alert2.hashCode());
    }
}