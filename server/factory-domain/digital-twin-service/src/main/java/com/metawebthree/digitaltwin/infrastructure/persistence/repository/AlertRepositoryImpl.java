package com.metawebthree.digitaltwin.infrastructure.persistence.repository;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.metawebthree.digitaltwin.domain.entity.Alert;
import com.metawebthree.digitaltwin.domain.repository.AlertRepository;
import com.metawebthree.digitaltwin.infrastructure.persistence.converter.AlertConverter;
import com.metawebthree.digitaltwin.infrastructure.persistence.dataobject.AlertDO;
import com.metawebthree.digitaltwin.infrastructure.persistence.mapper.AlertMapper;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Repository
public class AlertRepositoryImpl implements AlertRepository {

    private final AlertMapper alertMapper;

    public AlertRepositoryImpl(AlertMapper alertMapper) {
        this.alertMapper = alertMapper;
    }

    @Override
    public Optional<Alert> findById(Long id) {
        return Optional.ofNullable(alertMapper.selectById(id))
                .map(AlertConverter::toEntity);
    }

    @Override
    public Optional<Alert> findByAlertCode(String alertCode) {
        AlertDO d = alertMapper.selectOne(
                new LambdaQueryWrapper<AlertDO>().eq(AlertDO::getAlertCode, alertCode));
        return Optional.ofNullable(AlertConverter.toEntity(d));
    }

    @Override
    public List<Alert> findByDeviceCode(String deviceCode) {
        return alertMapper.selectList(
                new LambdaQueryWrapper<AlertDO>().eq(AlertDO::getDeviceCode, deviceCode))
                .stream().map(AlertConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByWorkshopId(String workshopId) {
        return alertMapper.selectList(
                new LambdaQueryWrapper<AlertDO>().eq(AlertDO::getWorkshopId, workshopId))
                .stream().map(AlertConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByStatus(Alert.AlertStatus status) {
        return alertMapper.selectList(
                new LambdaQueryWrapper<AlertDO>().eq(AlertDO::getStatus, status.name()))
                .stream().map(AlertConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findByLevel(Alert.AlertLevel level) {
        return alertMapper.selectList(
                new LambdaQueryWrapper<AlertDO>().eq(AlertDO::getLevel, level.name()))
                .stream().map(AlertConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public List<Alert> findAll() {
        return alertMapper.selectList(null)
                .stream().map(AlertConverter::toEntity).collect(Collectors.toList());
    }

    @Override
    public Alert save(Alert alert) {
        AlertDO d = AlertConverter.toDO(alert);
        alertMapper.insert(d);
        alert.setId(d.getId());
        return alert;
    }

    @Override
    public void update(Alert alert) {
        AlertDO d = AlertConverter.toDO(alert);
        alertMapper.updateById(d);
    }

    @Override
    public void deleteById(Long id) {
        alertMapper.deleteById(id);
    }
}
