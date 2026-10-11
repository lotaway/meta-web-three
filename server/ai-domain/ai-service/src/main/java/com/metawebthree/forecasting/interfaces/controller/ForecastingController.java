package com.metawebthree.forecasting.interfaces.controller;

import com.metawebthree.forecasting.application.command.ForecastingCommandService;
import com.metawebthree.forecasting.application.query.ForecastingQueryService;
import com.metawebthree.forecasting.domain.entity.SalesHistory;
import com.metawebthree.forecasting.domain.repository.SalesHistoryRepository;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@RestController
@RequestMapping("/api/forecasting")
public class ForecastingController {

    private final ForecastingCommandService commandService;
    private final ForecastingQueryService queryService;
    private final SalesHistoryRepository salesHistoryRepository;

    public ForecastingController(
            ForecastingCommandService commandService,
            ForecastingQueryService queryService,
            SalesHistoryRepository salesHistoryRepository) {
        this.commandService = commandService;
        this.queryService = queryService;
        this.salesHistoryRepository = salesHistoryRepository;
    }

    @PostMapping("/forecast")
    public ResponseEntity<Map<String, Object>> createForecast(
            @RequestBody Map<String, Object> request) {
        String skuCode = (String) request.get("skuCode");
        String skuName = (String) request.get("skuName");
        Long warehouseId = ((Number) request.get("warehouseId")).longValue();
        LocalDate forecastDate = LocalDate.parse((String) request.get("forecastDate"));
        Integer quantity = (Integer) request.get("quantity");
        String modelName = (String) request.get("modelName");

        Long forecastId = commandService.createForecast(
            skuCode, skuName, warehouseId, forecastDate, quantity, modelName);

        return ResponseEntity.ok(Map.of("forecastId", forecastId));
    }

    @PostMapping("/forecast/algorithm")
    public ResponseEntity<Map<String, Object>> createForecastWithAlgorithm(
            @RequestBody Map<String, Object> request) {
        String skuCode = (String) request.get("skuCode");
        String skuName = (String) request.get("skuName");
        Long warehouseId = ((Number) request.get("warehouseId")).longValue();
        LocalDate forecastDate = LocalDate.parse((String) request.get("forecastDate"));
        String algorithm = (String) request.getOrDefault("algorithm", "SMA");
        Integer windowSize = (Integer) request.getOrDefault("windowSize", 7);

        Long forecastId = commandService.createForecastWithAlgorithm(
            skuCode, skuName, warehouseId, forecastDate, algorithm, windowSize);

        return ResponseEntity.ok(Map.of("forecastId", forecastId));
    }

    @PostMapping("/sales-history/sample")
    public ResponseEntity<Map<String, Object>> generateSampleSalesHistory(
            @RequestBody Map<String, Object> request) {
        String skuCode = (String) request.get("skuCode");
        Long warehouseId = ((Number) request.get("warehouseId")).longValue();
        Integer days = (Integer) request.getOrDefault("days", 90);
        Integer baseQuantity = (Integer) request.getOrDefault("baseQuantity", 100);

        List<SalesHistory> salesHistoryList = IntStream.range(0, days)
            .mapToObj(i -> {
                LocalDate date = LocalDate.now().minusDays(days - i - 1);
                int quantity = baseQuantity + (int) (Math.random() * 40 - 20)
                    + (int) (Math.sin(i / 7.0) * 10);
                return new SalesHistory(skuCode, warehouseId, date, Math.max(0, quantity));
            })
            .collect(Collectors.toList());

        salesHistoryRepository.saveBatch(salesHistoryList);

        return ResponseEntity.ok(Map.of(
            "message", "Sample sales history generated",
            "skuCode", skuCode,
            "warehouseId", warehouseId,
            "recordCount", salesHistoryList.size()
        ));
    }

    @PostMapping("/forecast/{id}/confirm")
    public ResponseEntity<Void> confirmForecast(@PathVariable Long id) {
        commandService.confirmForecast(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/forecast/{id}/adjust")
    public ResponseEntity<Void> adjustForecast(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        Integer newQuantity = (Integer) request.get("newQuantity");
        commandService.adjustForecast(id, newQuantity);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/forecast/{id}/record-actual")
    public ResponseEntity<Void> recordActualSales(
            @PathVariable Long id,
            @RequestBody Map<String, Object> request) {
        Integer actualQuantity = (Integer) request.get("actualQuantity");
        commandService.recordActualSales(id, actualQuantity);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/forecast/{id}")
    public ResponseEntity<?> getForecast(@PathVariable Long id) {
        return queryService.getForecastById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/forecast/sku/{skuCode}")
    public ResponseEntity<?> getForecastBySku(@PathVariable String skuCode) {
        return ResponseEntity.ok(queryService.getForecastBySkuCode(skuCode));
    }

    @GetMapping("/forecast/warehouse/{warehouseId}")
    public ResponseEntity<?> getForecastByWarehouse(@PathVariable Long warehouseId) {
        return ResponseEntity.ok(queryService.getForecastByWarehouse(warehouseId));
    }

    @GetMapping("/forecast/history")
    public ResponseEntity<?> getForecastHistory(
            @RequestParam String skuCode,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {
        return ResponseEntity.ok(queryService.getForecastHistory(skuCode, startDate, endDate));
    }

    @PostMapping("/model")
    public ResponseEntity<Map<String, Object>> createModel(
            @RequestBody Map<String, Object> request) {
        String modelName = (String) request.get("modelName");
        String modelType = (String) request.get("modelType");
        String algorithm = (String) request.get("algorithm");
        String featureConfig = (String) request.get("featureConfig");
        Integer trainingDays = (Integer) request.get("trainingDays");

        Long modelId = commandService.createModel(
            modelName, modelType, algorithm, featureConfig, trainingDays);

        return ResponseEntity.ok(Map.of("modelId", modelId));
    }

    @PostMapping("/model/{id}/train")
    public ResponseEntity<Void> trainModel(@PathVariable Long id) {
        commandService.trainModel(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/model/{id}/deploy")
    public ResponseEntity<Void> deployModel(@PathVariable Long id) {
        commandService.deployModel(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/model/{id}")
    public ResponseEntity<?> getModel(@PathVariable Long id) {
        return queryService.getModelById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/model")
    public ResponseEntity<?> getAllModels() {
        return ResponseEntity.ok(queryService.getAllModels());
    }
}
