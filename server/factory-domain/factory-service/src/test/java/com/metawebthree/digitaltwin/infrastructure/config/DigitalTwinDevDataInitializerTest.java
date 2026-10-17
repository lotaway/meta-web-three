package com.metawebthree.digitaltwin.infrastructure.config;

import com.metawebthree.digitaltwin.application.command.DigitalTwinCommandService;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.entity.ProductionLine;
import com.metawebthree.digitaltwin.domain.entity.Workshop;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.domain.repository.ProductionLineRepository;
import com.metawebthree.digitaltwin.domain.repository.WorkshopRepository;
import org.junit.jupiter.api.Test;
import org.springframework.boot.CommandLineRunner;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DigitalTwinDevDataInitializerTest {

    private final WorkshopRepository workshops = mock(WorkshopRepository.class);
    private final ProductionLineRepository lines = mock(ProductionLineRepository.class);
    private final DeviceRepository devices = mock(DeviceRepository.class);
    private final AlertRepository alerts = mock(AlertRepository.class);
    private final DigitalTwinCommandService commandService = mock(DigitalTwinCommandService.class);

    private CommandLineRunner runner() {
        return new DigitalTwinDevDataInitializer()
                .seedDigitalTwinDemoData(devices, workshops, lines, alerts, commandService);
    }

    @Test
    void seedsEveryDemoEntityWhenAbsent() throws Exception {
        when(workshops.findByWorkshopCode("WS-01")).thenReturn(Optional.empty());
        when(lines.findByLineCode("LINE-01")).thenReturn(Optional.empty());
        when(devices.findByDeviceCode(anyString())).thenReturn(Optional.empty());
        when(alerts.findByDeviceCode(anyString())).thenReturn(List.of());

        runner().run();

        verify(commandService).createWorkshop("WS-01", "一号车间", "数字孪生演示车间");
        verify(commandService).createProductionLine("LINE-01", "演示产线", "WS-01", 100);
        verify(commandService).updateProductionLineOutput("LINE-01", 85);
        verify(commandService, times(8)).registerDevice(anyString(), anyString(), anyString(), eq("WS-01"), eq("LINE-01"));
        verify(commandService, times(8)).updateDevicePosition(anyString(), anyDouble(), anyDouble(), anyDouble(), anyDouble());
        verify(commandService, times(8)).updateDeviceStatus(anyString(), any());
        verify(commandService, times(3)).createAlert(anyString(), eq("WS-01"), any(), any(), anyString(), anyString());
    }

    @Test
    void skipsEveryDemoEntityWhenPresent() throws Exception {
        when(workshops.findByWorkshopCode("WS-01")).thenReturn(Optional.of(new Workshop()));
        when(lines.findByLineCode("LINE-01")).thenReturn(Optional.of(new ProductionLine()));
        when(devices.findByDeviceCode(anyString())).thenReturn(Optional.of(new Device()));
        when(alerts.findByDeviceCode(anyString())).thenReturn(List.of(new Alert()));

        runner().run();

        verifyNoInteractions(commandService);
    }

    @Test
    void doesNotRecreateWorkshopWhenExistingButDevicesEmpty() throws Exception {
        when(workshops.findByWorkshopCode("WS-01")).thenReturn(Optional.of(new Workshop()));
        when(lines.findByLineCode("LINE-01")).thenReturn(Optional.empty());
        when(devices.findByDeviceCode(anyString())).thenReturn(Optional.empty());
        when(alerts.findByDeviceCode(anyString())).thenReturn(List.of());

        runner().run();

        verify(commandService, never()).createWorkshop(anyString(), anyString(), anyString());
        verify(commandService).createProductionLine("LINE-01", "演示产线", "WS-01", 100);
        verify(commandService, times(8)).registerDevice(anyString(), anyString(), anyString(), eq("WS-01"), eq("LINE-01"));
        verify(commandService, times(3)).createAlert(anyString(), eq("WS-01"), any(), any(), anyString(), anyString());
    }

    @Test
    void seedFailureDoesNotBlockStartup() {
        when(workshops.findByWorkshopCode("WS-01")).thenReturn(Optional.empty());
        doThrow(new RuntimeException("Send failed"))
                .when(commandService).createWorkshop(anyString(), anyString(), anyString());

        assertDoesNotThrow(() -> runner().run());
    }
}
