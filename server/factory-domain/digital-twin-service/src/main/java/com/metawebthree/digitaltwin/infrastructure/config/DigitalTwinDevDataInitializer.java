package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;

@Configuration
@Profile("dev")
public class DigitalTwinDevDataInitializer {

    @Bean
    CommandLineRunner seedDigitalTwinDemoData(
            DeviceRepository deviceRepository,
            DigitalTwinCommandService commandService) {
        return args -> {
            if (!deviceRepository.findAll().isEmpty()) {
                return;
            }

            commandService.createWorkshop("WS-01", "一号车间", "数字孪生演示车间");
            commandService.createProductionLine("LINE-01", "演示产线", "WS-01", 100);

            registerDemoDevice(commandService, "AGV-001", "搬运机器人A1", "AGV", 2.0, 0.25, 3.0, 0.0, Device.DeviceStatus.RUNNING);
            registerDemoDevice(commandService, "AGV-002", "搬运机器人A2", "AGV", -3.0, 0.25, 5.0, 0.785, Device.DeviceStatus.RUNNING);
            registerDemoDevice(commandService, "ROBOT-001", "机械臂R1", "ROBOT", 5.0, 0.75, -2.0, 3.14159, Device.DeviceStatus.RUNNING);
            registerDemoDevice(commandService, "PLC-001", "PLC控制器C1", "PLC", -5.0, 0.3, -3.0, 0.0, Device.DeviceStatus.ONLINE);
            registerDemoDevice(commandService, "CONVEYOR-001", "传送带S1", "CONVEYOR", 0.0, 0.15, 0.0, 0.0, Device.DeviceStatus.RUNNING);
            registerDemoDevice(commandService, "AGV-003", "搬运机器人A3", "AGV", -2.0, 0.25, -4.0, 1.5708, Device.DeviceStatus.IDLE);
            registerDemoDevice(commandService, "ROBOT-002", "机械臂R2", "ROBOT", 7.0, 0.75, 2.0, -0.785, Device.DeviceStatus.WARNING);
            registerDemoDevice(commandService, "PLC-002", "PLC控制器C2", "PLC", -7.0, 0.3, 4.0, 0.0, Device.DeviceStatus.ERROR);

            commandService.createAlert(
                "PLC-002", "WS-01",
                Alert.AlertLevel.CRITICAL,
                Alert.AlertType.DEVICE_ERROR,
                "设备故障",
                "PLC控制器C2通信异常"
            );
            commandService.createAlert(
                "ROBOT-002", "WS-01",
                Alert.AlertLevel.WARNING,
                Alert.AlertType.TEMPERATURE_HIGH,
                "温度告警",
                "机械臂R2温度过高"
            );
            commandService.createAlert(
                "AGV-001", "WS-01",
                Alert.AlertLevel.INFO,
                Alert.AlertType.MAINTENANCE_DUE,
                "维护提醒",
                "搬运机器人A1即将到期维护"
            );

            commandService.updateProductionLineOutput("LINE-01", 85);
        };
    }

    private static void registerDemoDevice(
            DigitalTwinCommandService commandService,
            String code,
            String name,
            String type,
            double x,
            double y,
            double z,
            double rotation,
            Device.DeviceStatus status) {
        commandService.registerDevice(code, name, type, "WS-01", "LINE-01");
        commandService.updateDevicePosition(code, x, y, z, rotation);
        commandService.updateDeviceStatus(code, status);
    }
}
