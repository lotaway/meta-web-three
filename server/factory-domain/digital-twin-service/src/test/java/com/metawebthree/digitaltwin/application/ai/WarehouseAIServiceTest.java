package com.metawebthree.digitaltwin.application.ai;

import com.metawebthree.digitaltwin.infrastructure.client.AnomalyDetectionClient;
import com.metawebthree.digitaltwin.infrastructure.client.ForecastingServiceClient;
import com.metawebthree.digitaltwin.infrastructure.client.LocationRecommendationClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class WarehouseAIServiceTest {

    @Mock
    private ForecastingServiceClient forecastingClient;

    @Mock
    private LocationRecommendationClient locationClient;

    @Mock
    private AnomalyDetectionClient anomalyClient;

    private WarehouseAIService service;

    @BeforeEach
    void setUp() {
        service = new WarehouseAIService(forecastingClient, locationClient, anomalyClient);
    }

    // ==================== Demand Forecasting Tests ====================

    @Test
    void forecastDemand_shouldReturnResult_whenAIClientReturnsValidData() {
        ForecastingServiceClient.ForecastResult mockResult = new ForecastingServiceClient.ForecastResult();
        mockResult.setForecastId(1L);
        mockResult.setSkuCode("SKU-001");
        mockResult.setPredictedQuantity(100);
        mockResult.setConfidence(0.95);
        mockResult.setModelName("arima");

        when(forecastingClient.createForecast(anyString(), anyString(), anyLong(), any(LocalDate.class)))
            .thenReturn(mockResult);

        WarehouseAIService.ForecastingResult result = service.forecastDemand(
            "SKU-001", "Test Product", 1L, LocalDate.now().plusDays(7));

        assertNotNull(result);
        assertEquals("SKU-001", result.getSkuCode());
        assertEquals(100, result.getPredictedQuantity());
        assertEquals(0.95, result.getConfidence());
        assertEquals("arima", result.getModelName());
    }

    @Test
    void forecastDemand_shouldReturnNull_whenAIClientReturnsNull() {
        when(forecastingClient.createForecast(anyString(), anyString(), anyLong(), any(LocalDate.class)))
            .thenReturn(null);

        WarehouseAIService.ForecastingResult result = service.forecastDemand(
            "SKU-001", "Test Product", 1L, LocalDate.now().plusDays(7));

        assertNull(result);
    }

    @Test
    void getForecastHistory_shouldReturnMap_whenAIClientReturnsData() {
        Map<String, Object> mockHistory = Map.of(
            "forecasts", List.of(Map.of("date", "2024-01-01", "quantity", 100)),
            "total", 1
        );
        when(forecastingClient.getForecastHistory(anyString(), any(LocalDate.class), any(LocalDate.class)))
            .thenReturn(mockHistory);

        Map<String, Object> result = service.getForecastHistory(
            "SKU-001", LocalDate.now().minusDays(30), LocalDate.now());

        assertNotNull(result);
        assertEquals(1, result.get("total"));
    }

    @Test
    void getWarehouseForecasts_shouldReturnMap() {
        Map<String, Object> mockForecasts = Map.of("forecasts", List.of(), "count", 0);
        when(forecastingClient.getForecastByWarehouse(anyLong())).thenReturn(mockForecasts);

        Map<String, Object> result = service.getWarehouseForecasts(1L);

        assertNotNull(result);
    }

    // ==================== Location Recommendation Tests ====================

    @Test
    void recommendLocation_shouldReturnResult_whenAIClientReturnsValidData() {
        LocationRecommendationClient.LocationRecommendation mockRec = 
            new LocationRecommendationClient.LocationRecommendation();
        mockRec.setSkuCode("SKU-001");
        mockRec.setRecommendedZone("A");
        mockRec.setRecommendedShelf("A-01");
        mockRec.setRecommendedBin("A-01-01");
        mockRec.setScore(0.95);
        mockRec.setReason("High turnover + correlation with SKU-002");

        when(locationClient.recommendLocation(anyString(), anyLong(), anyInt()))
            .thenReturn(mockRec);

        WarehouseAIService.LocationRecResult result = service.recommendLocation(
            "SKU-001", 1L, 50);

        assertNotNull(result);
        assertEquals("A", result.getRecommendedZone());
        assertEquals("A-01", result.getRecommendedShelf());
        assertEquals(0.95, result.getScore());
    }

    @Test
    void recommendLocation_shouldReturnNull_whenAIClientReturnsNull() {
        when(locationClient.recommendLocation(anyString(), anyLong(), anyInt()))
            .thenReturn(null);

        WarehouseAIService.LocationRecResult result = service.recommendLocation(
            "SKU-001", 1L, 50);

        assertNull(result);
    }

    @Test
    void recommendBatchLocations_shouldReturnList() {
        LocationRecommendationClient.LocationRecommendation mockRec = 
            new LocationRecommendationClient.LocationRecommendation();
        mockRec.setSkuCode("SKU-001");
        mockRec.setRecommendedZone("A");
        mockRec.setScore(0.9);

        List<Map<String, Object>> items = List.of(
            Map.of("skuCode", "SKU-001", "quantity", 50),
            Map.of("skuCode", "SKU-002", "quantity", 30)
        );

        when(locationClient.recommendBatchLocations(anyList(), anyLong()))
            .thenReturn(List.of(mockRec));

        List<WarehouseAIService.LocationRecResult> results = 
            service.recommendBatchLocations(items, 1L);

        assertNotNull(results);
        assertEquals(1, results.size());
    }

    @Test
    void getProductCorrelations_shouldReturnMap() {
        Map<String, Double> mockCorrelations = Map.of("SKU-002", 0.85, "SKU-003", 0.72);
        when(locationClient.getProductCorrelations(anyString(), anyLong()))
            .thenReturn(mockCorrelations);

        Map<String, Double> result = service.getProductCorrelations("SKU-001", 1L);

        assertNotNull(result);
        assertEquals(0.85, result.get("SKU-002"));
    }

    @Test
    void getTurnoverAnalysis_shouldReturnMap() {
        Map<String, Object> mockAnalysis = Map.of("turnoverRate", 5.2, "rank", "A");
        when(locationClient.getTurnoverAnalysis(anyString(), anyLong()))
            .thenReturn(mockAnalysis);

        Map<String, Object> result = service.getTurnoverAnalysis("SKU-001", 1L);

        assertNotNull(result);
        assertEquals(5.2, result.get("turnoverRate"));
    }

    // ==================== Anomaly Detection Tests ====================

    @Test
    void detectAnomalies_shouldReturnList_whenAIClientReturnsData() {
        AnomalyDetectionClient.AnomalyResult mockAnomaly = new AnomalyDetectionClient.AnomalyResult();
        mockAnomaly.setAnomalyId(1L);
        mockAnomaly.setAnomalyType("SPIKE");
        mockAnomaly.setSeverity("HIGH");
        mockAnomaly.setDescription("Unusual inventory spike detected");
        mockAnomaly.setTimestamp(LocalDateTime.now());
        mockAnomaly.setSkuCode("SKU-001");
        mockAnomaly.setWarehouseId(1L);
        mockAnomaly.setConfidence(0.92);
        mockAnomaly.setRecommendedAction("Investigate inventory source");

        when(anomalyClient.detectAnomalies(anyString(), anyLong(), anyInt()))
            .thenReturn(List.of(mockAnomaly));

        List<WarehouseAIService.AnomalyResult> results = 
            service.detectAnomalies("SKU-001", 1L, 24);

        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals("SPIKE", results.get(0).getAnomalyType());
        assertEquals("HIGH", results.get(0).getSeverity());
        assertEquals(0.92, results.get(0).getConfidence());
    }

    @Test
    void detectAnomalies_shouldReturnEmptyList_whenAIClientReturnsEmpty() {
        when(anomalyClient.detectAnomalies(anyString(), anyLong(), anyInt()))
            .thenReturn(List.of());

        List<WarehouseAIService.AnomalyResult> results = 
            service.detectAnomalies("SKU-001", 1L, 24);

        assertNotNull(results);
        assertTrue(results.isEmpty());
    }

    @Test
    void detectSensorAnomalies_shouldReturnList() {
        AnomalyDetectionClient.AnomalyResult mockAnomaly = new AnomalyDetectionClient.AnomalyResult();
        mockAnomaly.setAnomalyId(2L);
        mockAnomaly.setAnomalyType("TEMPERATURE_SPIKE");
        mockAnomaly.setSeverity("CRITICAL");
        mockAnomaly.setDescription("Temperature exceeds threshold");
        mockAnomaly.setTimestamp(LocalDateTime.now());
        mockAnomaly.setConfidence(0.98);

        List<Map<String, Object>> sensorData = List.of(
            Map.of("timestamp", "2024-01-01T10:00:00", "value", 85.0)
        );

        when(anomalyClient.detectSensorAnomalies(anyList()))
            .thenReturn(List.of(mockAnomaly));

        List<WarehouseAIService.AnomalyResult> results = 
            service.detectSensorAnomalies(sensorData);

        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals("CRITICAL", results.get(0).getSeverity());
    }

    @Test
    void detectInventoryAnomalies_shouldReturnList() {
        AnomalyDetectionClient.AnomalyResult mockAnomaly = new AnomalyDetectionClient.AnomalyResult();
        mockAnomaly.setAnomalyId(3L);
        mockAnomaly.setAnomalyType("NEGATIVE_STOCK");
        mockAnomaly.setSeverity("HIGH");

        when(anomalyClient.detectInventoryAnomalies(anyLong(), any(LocalDateTime.class), any(LocalDateTime.class)))
            .thenReturn(List.of(mockAnomaly));

        List<WarehouseAIService.AnomalyResult> results = 
            service.detectInventoryAnomalies(1L, 
                LocalDateTime.now().minusHours(24), 
                LocalDateTime.now());

        assertNotNull(results);
        assertEquals(1, results.size());
    }

    @Test
    void getActiveAlerts_shouldReturnList() {
        AnomalyDetectionClient.AnomalyResult mockAlert = new AnomalyDetectionClient.AnomalyResult();
        mockAlert.setAnomalyId(4L);
        mockAlert.setSeverity("MEDIUM");

        when(anomalyClient.getActiveAlerts(anyLong(), anyString()))
            .thenReturn(List.of(mockAlert));

        List<WarehouseAIService.AnomalyResult> results = 
            service.getActiveAlerts(1L, "MEDIUM");

        assertNotNull(results);
        assertEquals(1, results.size());
    }

    @Test
    void predictAnomalies_shouldReturnList() {
        AnomalyDetectionClient.AnomalyResult mockPrediction = new AnomalyDetectionClient.AnomalyResult();
        mockPrediction.setAnomalyId(5L);
        mockPrediction.setAnomalyType("PREDICTED_DEMAND_SPIKE");
        mockPrediction.setSeverity("MEDIUM");
        mockPrediction.setConfidence(0.78);

        when(anomalyClient.predictAnomalies(anyString(), anyLong(), anyInt()))
            .thenReturn(List.of(mockPrediction));

        List<WarehouseAIService.AnomalyResult> results = 
            service.predictAnomalies("SKU-001", 1L, 72);

        assertNotNull(results);
        assertEquals(1, results.size());
        assertEquals("PREDICTED_DEMAND_SPIKE", results.get(0).getAnomalyType());
    }
}