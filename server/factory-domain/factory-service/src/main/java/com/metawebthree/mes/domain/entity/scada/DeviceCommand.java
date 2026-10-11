package com.metawebthree.mes.domain.entity.scada;

import java.time.LocalDateTime;

public class DeviceCommand {
    private Long id;
    private String commandCode;
    private String equipmentCode;
    private CommandType commandType;
    private String payload;
    private CommandStatus status;
    private String createdBy;
    private LocalDateTime createdAt;
    private LocalDateTime executedAt;
    private String resultMessage;

    public enum CommandType {
        START, STOP, RESET, SET_PARAMETER, CALIBRATE, SET_SPEED, SET_TEMPERATURE, CUSTOM
    }

    public enum CommandStatus {
        PENDING, SENT, DELIVERED, EXECUTED, FAILED, TIMEOUT
    }

    public void create(String commandCode, String equipmentCode, CommandType commandType, String payload, String createdBy) {
        this.commandCode = commandCode;
        this.equipmentCode = equipmentCode;
        this.commandType = commandType;
        this.payload = payload;
        this.status = CommandStatus.PENDING;
        this.createdBy = createdBy;
        this.createdAt = LocalDateTime.now();
    }

    public void markSent() { this.status = CommandStatus.SENT; }
    public void markDelivered() { this.status = CommandStatus.DELIVERED; }
    public void markExecuted(String result) {
        this.status = CommandStatus.EXECUTED;
        this.executedAt = LocalDateTime.now();
        this.resultMessage = result;
    }
    public void markFailed(String error) {
        this.status = CommandStatus.FAILED;
        this.executedAt = LocalDateTime.now();
        this.resultMessage = error;
    }
    public void markTimeout() { this.status = CommandStatus.TIMEOUT; }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getCommandCode() { return commandCode; }
    public void setCommandCode(String commandCode) { this.commandCode = commandCode; }
    public String getEquipmentCode() { return equipmentCode; }
    public void setEquipmentCode(String equipmentCode) { this.equipmentCode = equipmentCode; }
    public CommandType getCommandType() { return commandType; }
    public void setCommandType(CommandType commandType) { this.commandType = commandType; }
    public String getPayload() { return payload; }
    public void setPayload(String payload) { this.payload = payload; }
    public CommandStatus getStatus() { return status; }
    public void setStatus(CommandStatus status) { this.status = status; }
    public String getCreatedBy() { return createdBy; }
    public void setCreatedBy(String createdBy) { this.createdBy = createdBy; }
    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }
    public LocalDateTime getExecutedAt() { return executedAt; }
    public void setExecutedAt(LocalDateTime executedAt) { this.executedAt = executedAt; }
    public String getResultMessage() { return resultMessage; }
    public void setResultMessage(String resultMessage) { this.resultMessage = resultMessage; }
}
