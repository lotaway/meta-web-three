package com.metawebthree.inventory.interfaces.controller.alert;

import com.metawebthree.inventory.domain.entity.alert.InventoryAlert;
import com.metawebthree.inventory.domain.entity.alert.InventoryAlertConfig;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertConfigRepository;
import com.metawebthree.inventory.domain.repository.alert.InventoryAlertRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/inventory/alert")
@RequiredArgsConstructor
@Slf4j
public class InventoryAlertController {

    private final InventoryAlertRepository alertRepository;
    private final InventoryAlertConfigRepository configRepository;

    @GetMapping("/active")
    public ResponseEntity<List<InventoryAlert>> getActiveAlerts() {
        List<InventoryAlert> alerts = alertRepository.findActiveAlerts();
        return ResponseEntity.ok(alerts);
    }

    @GetMapping("/sku/{skuCode}")
    public ResponseEntity<List<InventoryAlert>> getAlertsBySkuCode(@PathVariable String skuCode) {
        List<InventoryAlert> alerts = alertRepository.findBySkuCodeAndStatus(skuCode, null);
        return ResponseEntity.ok(alerts);
    }

    @GetMapping("/{id}")
    public ResponseEntity<InventoryAlert> getAlertById(@PathVariable Long id) {
        InventoryAlert alert = alertRepository.findById(id);
        if (alert == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(alert);
    }

    @PostMapping("/{id}/acknowledge")
    public ResponseEntity<Void> acknowledgeAlert(@PathVariable Long id, @RequestParam String userId) {
        InventoryAlert alert = alertRepository.findById(id);
        if (alert == null) {
            return ResponseEntity.notFound().build();
        }
        alert.acknowledge(userId);
        alertRepository.save(alert);
        log.info("预警已确认: alertCode={}, userId={}", alert.getAlertCode(), userId);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/resolve")
    public ResponseEntity<Void> resolveAlert(@PathVariable Long id, 
                                              @RequestParam String userId,
                                              @RequestParam String solution) {
        InventoryAlert alert = alertRepository.findById(id);
        if (alert == null) {
            return ResponseEntity.notFound().build();
        }
        alert.resolve(userId, solution);
        alertRepository.save(alert);
        log.info("预警已解决: alertCode={}, userId={}", alert.getAlertCode(), userId);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/config")
    public ResponseEntity<List<InventoryAlertConfig>> getAllConfigs() {
        List<InventoryAlertConfig> configs = configRepository.findAll();
        return ResponseEntity.ok(configs);
    }

    @GetMapping("/config/enabled")
    public ResponseEntity<List<InventoryAlertConfig>> getEnabledConfigs() {
        List<InventoryAlertConfig> configs = configRepository.findAllEnabled();
        return ResponseEntity.ok(configs);
    }

    @GetMapping("/config/{id}")
    public ResponseEntity<InventoryAlertConfig> getConfigById(@PathVariable Long id) {
        InventoryAlertConfig config = configRepository.findById(id);
        if (config == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(config);
    }

    @PostMapping("/config")
    public ResponseEntity<InventoryAlertConfig> createConfig(@RequestBody InventoryAlertConfig config) {
        InventoryAlertConfig saved = configRepository.save(config);
        log.info("创建预警配置成功: id={}, warehouseCode={}", saved.getId(), saved.getWarehouseCode());
        return ResponseEntity.ok(saved);
    }

    @PutMapping("/config/{id}")
    public ResponseEntity<InventoryAlertConfig> updateConfig(@PathVariable Long id, 
                                                               @RequestBody InventoryAlertConfig config) {
        InventoryAlertConfig existing = configRepository.findById(id);
        if (existing == null) {
            return ResponseEntity.notFound().build();
        }
        config.setId(id);
        InventoryAlertConfig saved = configRepository.save(config);
        log.info("更新预警配置成功: id={}", id);
        return ResponseEntity.ok(saved);
    }

    @DeleteMapping("/config/{id}")
    public ResponseEntity<Void> deleteConfig(@PathVariable Long id) {
        InventoryAlertConfig existing = configRepository.findById(id);
        if (existing == null) {
            return ResponseEntity.notFound().build();
        }
        configRepository.deleteById(id);
        log.info("删除预警配置成功: id={}", id);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/config/{id}/toggle")
    public ResponseEntity<InventoryAlertConfig> toggleConfig(@PathVariable Long id) {
        InventoryAlertConfig config = configRepository.findById(id);
        if (config == null) {
            return ResponseEntity.notFound().build();
        }
        config.setEnabled(!Boolean.TRUE.equals(config.getEnabled()));
        InventoryAlertConfig saved = configRepository.save(config);
        log.info("切换预警配置状态: id={}, enabled={}", id, saved.getEnabled());
        return ResponseEntity.ok(saved);
    }
}