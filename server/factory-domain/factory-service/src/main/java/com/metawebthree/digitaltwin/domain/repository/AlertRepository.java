package com.metawebthree.digitaltwin.domain.repository;

import com.metawebthree.digitaltwin.domain.entity.Alert;
import java.util.List;
import java.util.Optional;

public interface AlertRepository {
    Optional<Alert> findById(Long id);
    Optional<Alert> findByAlertCode(String alertCode);
    List<Alert> findByDeviceCode(String deviceCode);
    List<Alert> findByWorkshopId(String workshopId);
    List<Alert> findByStatus(Alert.AlertStatus status);
    List<Alert> findByLevel(Alert.AlertLevel level);
    List<Alert> findAll();
    Alert save(Alert alert);
    void update(Alert alert);
    void deleteById(Long id);
}