package com.metawebthree.digitaltwin.domain.service;

import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import java.util.List;

public interface DigitalTwinDomainService {
    // Device operations
    Device registerDevice(String deviceCode, String deviceName, String deviceType, 
                         String workshopId, String productionLineId);
    void updateDeviceStatus(String deviceCode, Device.DeviceStatus status);
    void deviceHeartbeat(String deviceCode);
    void updateDevicePosition(String deviceCode, Double x, Double y, Double z, Double rotation);
    List<Device> getWorkshopDevices(String workshopId);
    List<Device> getOnlineDevices();
    
    // Workshop operations
    Workshop createWorkshop(String workshopCode, String workshopName, String description);
    void updateWorkshopStatus(String workshopId, Workshop.WorkshopStatus status);
    List<Workshop> getAllWorkshops();
    
    // ProductionLine operations
    ProductionLine createProductionLine(String lineCode, String lineName, 
                                        String workshopId, Integer capacity);
    void updateProductionLineStatus(String lineCode, ProductionLine.ProductionLineStatus status);
    void updateProductionLineOutput(String lineCode, Integer output);
    List<ProductionLine> getWorkshopProductionLines(String workshopId);
    
    // Alert operations
    Alert createAlert(String deviceCode, String workshopId, Alert.AlertLevel level,
                     Alert.AlertType type, String title, String description);
    void acknowledgeAlert(Long alertId, String acknowledgedBy);
    void resolveAlert(Long alertId, String solution, String resolvedBy);
    List<Alert> getActiveAlerts();
    List<Alert> getDeviceAlerts(String deviceCode);
    
    // Statistics
    Long getOnlineDeviceCount();
    Long getActiveAlertCount();
    Double getAverageEfficiency();
}