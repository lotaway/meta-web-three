package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.metawebthree.digitaltwin.domain.entity.Device;
import com.metawebthree.digitaltwin.domain.repository.DeviceRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.DeviceConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.DeviceDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.DeviceMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class DeviceRepositoryImpl implements DeviceRepository {

    private final DeviceMapper deviceMapper;

    public DeviceRepositoryImpl(DeviceMapper deviceMapper) {
        this.deviceMapper = deviceMapper;
    }

    @Override
    public Optional<Device> findById(Long id) {
        return Optional.ofNullable(deviceMapper.selectById(id))
                .map(DeviceConverter::toEntity);
    }

    @Override
    public Optional<Device> findByDeviceCode(String deviceCode) {
        DeviceDO d = deviceMapper.selectOne(
                new LambdaQueryWrapper<DeviceDO>().eq(DeviceDO::getDeviceCode, deviceCode));
        return Optional.ofNullable(DeviceConverter.toEntity(d));
    }

    @Override
    public List<Device> findByWorkshopId(String workshopId) {
        return deviceMapper.selectList(
                new LambdaQueryWrapper<DeviceDO>().eq(DeviceDO::getWorkshopId, workshopId))
                .stream().map(DeviceConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Device> findByProductionLineId(String productionLineId) {
        return deviceMapper.selectList(
                new LambdaQueryWrapper<DeviceDO>().eq(DeviceDO::getProductionLineId, productionLineId))
                .stream().map(DeviceConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Device> findByStatus(Device.DeviceStatus status) {
        return deviceMapper.selectList(
                new LambdaQueryWrapper<DeviceDO>().eq(DeviceDO::getStatus, status.name()))
                .stream().map(DeviceConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Device> findAll() {
        return deviceMapper.selectList(null)
                .stream().map(DeviceConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public IPage<Device> findPaginated(int page, int size) {
        Page<DeviceDO> pageObj = new Page<>(page, size);
        IPage<DeviceDO> result = deviceMapper.selectPage(pageObj, null);
        return result.convert(DeviceConverter::toEntity);
    }

    @Override
    public Device save(Device device) {
        DeviceDO d = DeviceConverter.toDO(device);
        deviceMapper.insert(d);
        device.setId(d.getId());
        return device;
    }

    @Override
    public void update(Device device) {
        DeviceDO d = DeviceConverter.toDO(device);
        deviceMapper.updateById(d);
    }

    @Override
    public void deleteById(Long id) {
        deviceMapper.deleteById(id);
    }
}
