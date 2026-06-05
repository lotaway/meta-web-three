package com.metawebthree.mes.infrastructure.persistence.repository.scada;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand.CommandStatus;
import com.metawebthree.mes.domain.entity.scada.DeviceCommand.CommandType;
import com.metawebthree.mes.domain.repository.scada.DeviceCommandRepository;
import com.metawebthree.mes.infrastructure.persistence.dataobject.ScadaDeviceCommandDO;
import com.metawebthree.mes.infrastructure.persistence.mapper.ScadaDeviceCommandMapper;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class DeviceCommandRepositoryImpl implements DeviceCommandRepository {

    private final ScadaDeviceCommandMapper mapper;

    public DeviceCommandRepositoryImpl(ScadaDeviceCommandMapper mapper) {
        this.mapper = mapper;
    }

    @Override
    public Optional<DeviceCommand> findById(Long id) {
        return Optional.ofNullable(mapper.selectById(id)).map(this::toEntity);
    }

    @Override
    public Optional<DeviceCommand> findByCommandCode(String commandCode) {
        LambdaQueryWrapper<ScadaDeviceCommandDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaDeviceCommandDO::getCommandCode, commandCode);
        return Optional.ofNullable(mapper.selectOne(w)).map(this::toEntity);
    }

    @Override
    public List<DeviceCommand> findByEquipmentCode(String equipmentCode) {
        LambdaQueryWrapper<ScadaDeviceCommandDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaDeviceCommandDO::getEquipmentCode, equipmentCode)
            .orderByDesc(ScadaDeviceCommandDO::getCreatedAt);
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<DeviceCommand> findByStatus(CommandStatus status) {
        LambdaQueryWrapper<ScadaDeviceCommandDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaDeviceCommandDO::getStatus, status.name());
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<DeviceCommand> findByEquipmentCodeAndStatus(String equipmentCode, CommandStatus status) {
        LambdaQueryWrapper<ScadaDeviceCommandDO> w = new LambdaQueryWrapper<>();
        w.eq(ScadaDeviceCommandDO::getEquipmentCode, equipmentCode)
            .eq(ScadaDeviceCommandDO::getStatus, status.name());
        return mapper.selectList(w).stream().map(this::toEntity).collect(Collectors.toList());
    }

    @Override
    public DeviceCommand save(DeviceCommand entity) {
        ScadaDeviceCommandDO doObj = toDO(entity);
        if (doObj.getId() == null) {
            mapper.insert(doObj);
            entity.setId(doObj.getId());
        } else {
            mapper.updateById(doObj);
        }
        return entity;
    }

    @Override
    public void update(DeviceCommand entity) {
        if (entity.getId() != null) {
            mapper.updateById(toDO(entity));
        }
    }

    @Override
    public void deleteById(Long id) {
        mapper.deleteById(id);
    }

    private DeviceCommand toEntity(ScadaDeviceCommandDO doObj) {
        DeviceCommand entity = new DeviceCommand();
        entity.setId(doObj.getId());
        entity.setCommandCode(doObj.getCommandCode());
        entity.setEquipmentCode(doObj.getEquipmentCode());
        entity.setCommandType(doObj.getCommandType() != null ? CommandType.valueOf(doObj.getCommandType()) : null);
        entity.setPayload(doObj.getPayload());
        entity.setStatus(doObj.getStatus() != null ? CommandStatus.valueOf(doObj.getStatus()) : CommandStatus.PENDING);
        entity.setCreatedBy(doObj.getCreatedBy());
        entity.setCreatedAt(doObj.getCreatedAt());
        entity.setExecutedAt(doObj.getExecutedAt());
        entity.setResultMessage(doObj.getResultMessage());
        return entity;
    }

    private ScadaDeviceCommandDO toDO(DeviceCommand entity) {
        ScadaDeviceCommandDO doObj = new ScadaDeviceCommandDO();
        doObj.setId(entity.getId());
        doObj.setCommandCode(entity.getCommandCode());
        doObj.setEquipmentCode(entity.getEquipmentCode());
        doObj.setCommandType(entity.getCommandType() != null ? entity.getCommandType().name() : null);
        doObj.setPayload(entity.getPayload());
        doObj.setStatus(entity.getStatus() != null ? entity.getStatus().name() : CommandStatus.PENDING.name());
        doObj.setCreatedBy(entity.getCreatedBy());
        doObj.setCreatedAt(entity.getCreatedAt());
        doObj.setExecutedAt(entity.getExecutedAt());
        doObj.setResultMessage(entity.getResultMessage());
        return doObj;
    }
}
