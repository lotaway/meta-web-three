package com.metawebthree.mes.interfaces.controller;

import com.metawebthree.common.annotations.RequirePermission;
import com.metawebthree.mes.common.MesPermissions;
import com.metawebthree.mes.domain.entity.SpcControlChart;
import com.metawebthree.mes.domain.entity.SpcControlChart.ChartType;
import com.metawebthree.mes.domain.repository.SpcControlChartRepository;
import com.metawebthree.mes.interfaces.dto.SpcControlChartDTO;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/mes/qc/spc-control-chart")
public class SpcControlChartController {
    
    private final SpcControlChartRepository repository;
    
    public SpcControlChartController(SpcControlChartRepository repository) {
        this.repository = repository;
    }
    
    @GetMapping
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_READ)
    public ResponseEntity<List<SpcControlChartDTO>> getAll() {
        List<SpcControlChartDTO> dtos = repository.findAll().stream()
                .map(SpcControlChartDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/{id}")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_READ)
    public ResponseEntity<SpcControlChartDTO> getById(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> ResponseEntity.ok(SpcControlChartDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/code/{code}")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_READ)
    public ResponseEntity<SpcControlChartDTO> getByCode(@PathVariable String code) {
        return repository.findByChartCode(code)
                .map(entity -> ResponseEntity.ok(SpcControlChartDTO.fromEntity(entity)))
                .orElse(ResponseEntity.notFound().build());
    }
    
    @GetMapping("/type/{type}")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_READ)
    public ResponseEntity<List<SpcControlChartDTO>> getByChartType(@PathVariable String type) {
        ChartType chartType = ChartType.valueOf(type);
        List<SpcControlChartDTO> dtos = repository.findByChartType(chartType).stream()
                .map(SpcControlChartDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @GetMapping("/enabled")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_READ)
    public ResponseEntity<List<SpcControlChartDTO>> getEnabled() {
        List<SpcControlChartDTO> dtos = repository.findByIsEnabled(true).stream()
                .map(SpcControlChartDTO::fromEntity)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }
    
    @PostMapping
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_CREATE)
    public ResponseEntity<SpcControlChartDTO> create(@RequestBody Map<String, Object> request) {
        String chartCode = (String) request.get("chartCode");
        String chartName = (String) request.get("chartName");
        String chartTypeStr = (String) request.get("chartType");
        String parameterCode = (String) request.get("parameterCode");
        
        if (repository.existsByChartCode(chartCode)) {
            return ResponseEntity.badRequest().build();
        }
        
        SpcControlChart entity = new SpcControlChart();
        entity.create(chartCode, chartName, ChartType.valueOf(chartTypeStr), parameterCode);
        
        if (request.containsKey("limits")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> limitsMap = (Map<String, Object>) request.get("limits");
            entity.setControlLimits(
                getDouble(limitsMap, "usl"),
                getDouble(limitsMap, "lsl"),
                getDouble(limitsMap, "ucl"),
                getDouble(limitsMap, "lcl"),
                getDouble(limitsMap, "centerLine")
            );
        }
        
        if (request.containsKey("defaultAlarmRules") && Boolean.TRUE.equals(request.get("defaultAlarmRules"))) {
            entity.addDefaultAlarmRules();
        }
        
        if (request.containsKey("samplingConfig")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> configMap = (Map<String, Object>) request.get("samplingConfig");
            entity.setSamplingConfig(
                getInteger(configMap, "sampleSize"),
                getInteger(configMap, "frequencyMinutes"),
                getInteger(configMap, "subgroupCount"),
                (String) configMap.get("samplingMethod")
            );
        }
        
        SpcControlChart saved = repository.save(entity);
        return ResponseEntity.status(201).body(SpcControlChartDTO.fromEntity(saved));
    }
    
    @PutMapping("/{id}")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_UPDATE)
    public ResponseEntity<SpcControlChartDTO> update(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        
        return repository.findById(id)
                .map(entity -> {
                    if (request.containsKey("chartName")) {
                        entity.setChartName((String) request.get("chartName"));
                    }
                    if (request.containsKey("parameterCode")) {
                        entity.setParameterCode((String) request.get("parameterCode"));
                    }
                    if (request.containsKey("limits")) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> limitsMap = (Map<String, Object>) request.get("limits");
                        entity.setControlLimits(
                            getDouble(limitsMap, "usl"),
                            getDouble(limitsMap, "lsl"),
                            getDouble(limitsMap, "ucl"),
                            getDouble(limitsMap, "lcl"),
                            getDouble(limitsMap, "centerLine")
                        );
                    }
                    if (request.containsKey("samplingConfig")) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> configMap = (Map<String, Object>) request.get("samplingConfig");
                        entity.setSamplingConfig(
                            getInteger(configMap, "sampleSize"),
                            getInteger(configMap, "frequencyMinutes"),
                            getInteger(configMap, "subgroupCount"),
                            (String) configMap.get("samplingMethod")
                        );
                    }
                    
                    SpcControlChart saved = repository.save(entity);
                    return ResponseEntity.ok(SpcControlChartDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @DeleteMapping("/{id}")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_DELETE)
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (repository.findById(id).isPresent()) {
            repository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
    
    @PostMapping("/{id}/enable")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_UPDATE)
    public ResponseEntity<SpcControlChartDTO> enable(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.enable();
                    SpcControlChart saved = repository.save(entity);
                    return ResponseEntity.ok(SpcControlChartDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    @PostMapping("/{id}/disable")
    @RequirePermission(MesPermissions.QC_SPC_CONTROL_CHART_UPDATE)
    public ResponseEntity<SpcControlChartDTO> disable(@PathVariable Long id) {
        return repository.findById(id)
                .map(entity -> {
                    entity.disable();
                    SpcControlChart saved = repository.save(entity);
                    return ResponseEntity.ok(SpcControlChartDTO.fromEntity(saved));
                })
                .orElse(ResponseEntity.notFound().build());
    }
    
    private Double getDouble(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (value == null) return null;
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        return Double.parseDouble(value.toString());
    }
    
    private Integer getInteger(Map<String, Object> map, String key) {
        Object value = map.get(key);
        if (value == null) return null;
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        return Integer.parseInt(value.toString());
    }
}