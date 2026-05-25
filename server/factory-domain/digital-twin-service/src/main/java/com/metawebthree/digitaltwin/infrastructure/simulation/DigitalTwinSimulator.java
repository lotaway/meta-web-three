package com.metawebthree.digitaltwin.infrastructure.simulation;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Profile;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

@Component
@Profile("dev")
public class DigitalTwinSimulator {

    private static final Logger log = LoggerFactory.getLogger(DigitalTwinSimulator.class);

    private final DigitalTwinCommandService commandService;
    private final DigitalTwinWebSocketHandler webSocketHandler;
    private final DeviceRepository deviceRepository;
    private final ProductionLineRepository productionLineRepository;

    private final Random random = new Random();
    private final Map<String, Integer> agvRouteIndices = new ConcurrentHashMap<>();

    private static final String[][] AGV_ROUTES = {
        {"AGV-001", "-2,0.25,-3", "0,0.25,-3", "2,0.25,-3", "4,0.25,-3", "4,0.25,-1", "4,0.25,1", "2,0.25,3", "0,0.25,3", "-2,0.25,3", "-4,0.25,3", "-4,0.25,0", "-4,0.25,-3"},
        {"AGV-002", "3,0.25,-2", "5,0.25,-2", "7,0.25,-2", "7,0.25,0", "7,0.25,2", "5,0.25,2", "3,0.25,2", "1,0.25,1", "1,0.25,-1", "3,0.25,-2"},
        {"AGV-003", "-6,0.25,-1", "-4,0.25,-1", "-2,0.25,-1", "0,0.25,-1", "2,0.25,-1", "2,0.25,1", "0,0.25,1", "-2,0.25,1", "-4,0.25,1", "-6,0.25,1", "-6,0.25,-1"},
    };

    private static final Device.DeviceStatus[] STATUS_POOL = {
        Device.DeviceStatus.RUNNING, Device.DeviceStatus.RUNNING,
        Device.DeviceStatus.RUNNING, Device.DeviceStatus.IDLE,
        Device.DeviceStatus.WARNING, Device.DeviceStatus.RUNNING,
        Device.DeviceStatus.RUNNING, Device.DeviceStatus.IDLE,
        Device.DeviceStatus.ERROR,
    };

    public DigitalTwinSimulator(
            DigitalTwinCommandService commandService,
            DigitalTwinWebSocketHandler webSocketHandler,
            DeviceRepository deviceRepository,
            ProductionLineRepository productionLineRepository) {
        this.commandService = commandService;
        this.webSocketHandler = webSocketHandler;
        this.deviceRepository = deviceRepository;
        this.productionLineRepository = productionLineRepository;
    }

    @Scheduled(initialDelay = 5000, fixedDelay = 10000)
    public void simulateStatusChanges() {
        List<Device> devices = deviceRepository.findAll();
        if (devices.isEmpty()) return;

        devices.stream()
                .filter(d -> d.getStatus() != Device.DeviceStatus.OFFLINE
                        && d.getStatus() != Device.DeviceStatus.MAINTENANCE)
                .skip(random.nextInt(devices.size()))
                .findFirst()
                .ifPresent(device -> {
                    Device.DeviceStatus newStatus = STATUS_POOL[random.nextInt(STATUS_POOL.length)];
                    if (newStatus != device.getStatus()) {
                        try {
                            commandService.updateDeviceStatus(device.getDeviceCode(), newStatus);
                            log.info("[Sim] {}: {} → {}", device.getDeviceCode(), device.getStatus(), newStatus);
                        } catch (Exception e) {
                            log.warn("[Sim] Failed to update device status: {}", device.getDeviceCode(), e);
                        }
                    }
                });
    }

    @Scheduled(initialDelay = 3000, fixedDelay = 3000)
    public void simulateAGVMovement() {
        for (String[] route : AGV_ROUTES) {
            String deviceCode = route[0];
            Device device = deviceRepository.findByDeviceCode(deviceCode).orElse(null);
            if (device == null || device.getStatus() != Device.DeviceStatus.RUNNING) continue;

            int idx = agvRouteIndices.getOrDefault(deviceCode, 1);
            idx = (idx % (route.length - 1)) + 1;
            agvRouteIndices.put(deviceCode, idx);

            String[] parts = route[idx].split(",");
            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);
            double z = Double.parseDouble(parts[2]);

            int nextIdx = (idx % (route.length - 1)) + 1;
            String[] nextParts = route[nextIdx].split(",");
            double nextX = Double.parseDouble(nextParts[0]);
            double nextZ = Double.parseDouble(nextParts[2]);
            double rotation = Math.atan2(nextZ - z, nextX - x);

            commandService.updateDevicePosition(deviceCode, x, y, z, rotation);
        }
    }

    @Scheduled(initialDelay = 7000, fixedDelay = 5000)
    public void simulateProductionOutput() {
        List<ProductionLine> lines = productionLineRepository.findAll();
        for (ProductionLine line : lines) {
            int capacity = line.getCapacity() != null ? line.getCapacity() : 100;
            int output = Math.min(capacity,
                    (int) (capacity * (0.5 + random.nextDouble() * 0.45)));
            commandService.updateProductionLineOutput(line.getLineCode(), output);
        }
    }

    @Scheduled(initialDelay = 15000, fixedDelay = 20000)
    public void simulateRandomAlerts() {
        if (random.nextDouble() > 0.35) return;

        List<Device> devices = deviceRepository.findAll().stream()
                .filter(d -> d.getStatus() != Device.DeviceStatus.OFFLINE)
                .toList();
        if (devices.isEmpty()) return;

        Device device = devices.get(random.nextInt(devices.size()));

        Alert.AlertType[] types = {
            Alert.AlertType.TEMPERATURE_HIGH,
            Alert.AlertType.VIBRATION_ABNORMAL,
            Alert.AlertType.MAINTENANCE_DUE,
            Alert.AlertType.NETWORK_ERROR,
            Alert.AlertType.POWER_FAILURE,
        };
        String[] titles = {"温度过高", "振动异常", "需要维护", "网络波动", "电压不稳"};
        Alert.AlertLevel[] levels = {
            Alert.AlertLevel.WARNING, Alert.AlertLevel.WARNING,
            Alert.AlertLevel.INFO, Alert.AlertLevel.WARNING, Alert.AlertLevel.ERROR,
        };

        int pick = random.nextInt(types.length);
        String workshopId = device.getWorkshopId() != null ? device.getWorkshopId() : "WS-01";
        try {
            commandService.createAlert(
                    device.getDeviceCode(), workshopId,
                    levels[pick], types[pick],
                    titles[pick],
                    device.getDeviceName() + " " + titles[pick]
            );
            log.info("[Sim] Alert: {} @ {}", titles[pick], device.getDeviceCode());
        } catch (Exception e) {
            log.warn("[Sim] Failed to create alert: {}", device.getDeviceCode(), e);
        }
    }
}
