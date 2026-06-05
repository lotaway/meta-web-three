package com.metawebthree.mes.domain.repository.scada;

import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand.CommandStatus;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand.CommandType;
import java.util.List;
import java.util.Optional;

public interface DeviceCommandRepository {
    Optional<DeviceCommand> findById(Long id);
    Optional<DeviceCommand> findByCommandCode(String commandCode);
    List<DeviceCommand> findByEquipmentCode(String equipmentCode);
    List<DeviceCommand> findByStatus(CommandStatus status);
    List<DeviceCommand> findByEquipmentCodeAndStatus(String equipmentCode, CommandStatus status);
    DeviceCommand save(DeviceCommand command);
    void update(DeviceCommand command);
    void deleteById(Long id);
}
