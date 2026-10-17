package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

@Slf4j
@Configuration
@Profile("dev")
public class DigitalTwinDevDataInitializer {

    @Bean
    CommandLineRunner seedDigitalTwinDemoData(
            DeviceRepository deviceRepository,
            WorkshopRepository workshopRepository,
            ProductionLineRepository productionLineRepository,
            AlertRepository alertRepository,
            DigitalTwinCommandService commandService) {
        return args -> {
            try {
                seedWorkshop(workshopRepository, commandService);
                seedProductionLine(productionLineRepository, commandService);
                seedDevices(deviceRepository, commandService);
                seedAlerts(alertRepository, commandService);
            } catch (Exception e) {
                log.warn("数字孪生演示数据初始化已中止: {}", e.getMessage());
            }
        };
    }

    private static void seedWorkshop(
            WorkshopRepository workshopRepository, DigitalTwinCommandService commandService) {
        if (workshopRepository.findByWorkshopCode("WS-01").isPresent()) {
            return;
        }
        commandService.createWorkshop("WS-01", "一号车间", "数字孪生演示车间");
    }

    private static void seedProductionLine(
            ProductionLineRepository productionLineRepository, DigitalTwinCommandService commandService) {
        if (productionLineRepository.findByLineCode("LINE-01").isPresent()) {
            return;
        }
        commandService.createProductionLine("LINE-01", "演示产线", "WS-01", 100);
        commandService.updateProductionLineOutput("LINE-01", 85);
    }

    private static void seedDevices(
            DeviceRepository deviceRepository, DigitalTwinCommandService commandService) {
        registerDemoDevice(deviceRepository, commandService, "AGV-001", "搬运机器人A1", "AGV", 2.0, 0.25, 3.0, 0.0, Device.DeviceStatus.RUNNING);
        registerDemoDevice(deviceRepository, commandService, "AGV-002", "搬运机器人A2", "AGV", -3.0, 0.25, 5.0, 0.785, Device.DeviceStatus.RUNNING);
        registerDemoDevice(deviceRepository, commandService, "ROBOT-001", "机械臂R1", "ROBOT", 5.0, 0.75, -2.0, 3.14159, Device.DeviceStatus.RUNNING);
        registerDemoDevice(deviceRepository, commandService, "PLC-001", "PLC控制器C1", "PLC", -5.0, 0.3, -3.0, 0.0, Device.DeviceStatus.ONLINE);
        registerDemoDevice(deviceRepository, commandService, "CONVEYOR-001", "传送带S1", "CONVEYOR", 0.0, 0.15, 0.0, 0.0, Device.DeviceStatus.RUNNING);
        registerDemoDevice(deviceRepository, commandService, "AGV-003", "搬运机器人A3", "AGV", -2.0, 0.25, -4.0, 1.5708, Device.DeviceStatus.IDLE);
        registerDemoDevice(deviceRepository, commandService, "ROBOT-002", "机械臂R2", "ROBOT", 7.0, 0.75, 2.0, -0.785, Device.DeviceStatus.WARNING);
        registerDemoDevice(deviceRepository, commandService, "PLC-002", "PLC控制器C2", "PLC", -7.0, 0.3, 4.0, 0.0, Device.DeviceStatus.ERROR);
    }

    private static void seedAlerts(
            AlertRepository alertRepository, DigitalTwinCommandService commandService) {
        if (alertRepository.findByDeviceCode("PLC-002").isEmpty()) {
            commandService.createAlert("PLC-002", "WS-01", Alert.AlertLevel.CRITICAL,
                    Alert.AlertType.DEVICE_ERROR, "设备故障", "PLC控制器C2通信异常");
        }
        if (alertRepository.findByDeviceCode("ROBOT-002").isEmpty()) {
            commandService.createAlert("ROBOT-002", "WS-01", Alert.AlertLevel.WARNING,
                    Alert.AlertType.TEMPERATURE_HIGH, "温度告警", "机械臂R2温度过高");
        }
        if (alertRepository.findByDeviceCode("AGV-001").isEmpty()) {
            commandService.createAlert("AGV-001", "WS-01", Alert.AlertLevel.INFO,
                    Alert.AlertType.MAINTENANCE_DUE, "维护提醒", "搬运机器人A1即将到期维护");
        }
    }

    private static void registerDemoDevice(
            DeviceRepository deviceRepository,
            DigitalTwinCommandService commandService,
            String code,
            String name,
            String type,
            double x,
            double y,
            double z,
            double rotation,
            Device.DeviceStatus status) {
        if (deviceRepository.findByDeviceCode(code).isPresent()) {
            return;
        }
        commandService.registerDevice(code, name, type, "WS-01", "LINE-01");
        commandService.updateDevicePosition(code, x, y, z, rotation);
        commandService.updateDeviceStatus(code, status);
    }
}
