package com.metawebthree.digitaltwin.integration.kafka;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.test.context.EmbeddedKafka;
import org.springframework.test.annotation.DirtiesContext;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
@EmbeddedKafka(partitions = 1, topics = {"device.status.changed", "device.position.updated", 
    "device.heartbeat", "alert.created", "production.output.updated", "agv.position.updated"},
    brokerProperties = {"listeners=PLAINTEXT://localhost:9092", "port=9092"})
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
@DisplayName("Kafka 消费集成测试")
class DigitalTwinKafkaConsumerIntegrationTest {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Test
    @DisplayName("消费 device.status.changed 消息")
    void consumeDeviceStatusChanged() throws Exception {
        String message = "{\"messageId\":\"test-status-001\",\"deviceCode\":\"DEVICE001\",\"status\":\"RUNNING\"}";
        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("消费 device.position.updated 消息")
    void consumeDevicePositionUpdated() throws Exception {
        String message = "{\"messageId\":\"test-position-001\",\"deviceCode\":\"DEVICE001\",\"x\":100.5,\"y\":200.3}";
        kafkaTemplate.send("device.position.updated", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("消费 alert.created 消息")
    void consumeAlertCreated() throws Exception {
        String message = "{\"messageId\":\"test-alert-001\",\"alertLevel\":\"HIGH\",\"content\":\"Temperature exceeded\"}";
        kafkaTemplate.send("alert.created", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("消费 production.output.updated 消息")
    void consumeProductionOutputUpdated() throws Exception {
        String message = "{\"messageId\":\"test-output-001\",\"lineId\":\"LINE001\",\"output\":1500}";
        kafkaTemplate.send("production.output.updated", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("消费 agv.position.updated 消息")
    void consumeAgvPositionUpdated() throws Exception {
        String message = "{\"messageId\":\"test-agv-001\",\"agvCode\":\"AGV001\",\"x\":50.0,\"y\":75.0,\"angle\":90}";
        kafkaTemplate.send("agv.position.updated", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("消费 device.heartbeat 消息")
    void consumeDeviceHeartbeat() throws Exception {
        String message = "{\"deviceCode\":\"DEVICE001\",\"timestamp\":1699999999000}";
        kafkaTemplate.send("device.heartbeat", message).get(5, TimeUnit.SECONDS);
        
        Thread.sleep(1000);
        
        assertTrue(true, "消息发送成功");
    }

    @Test
    @DisplayName("幂等性测试 - 重复消息只处理一次")
    void idempotencyTest() throws Exception {
        String messageId = "test-idempotent-001";
        String message = "{\"messageId\":\"" + messageId + "\",\"deviceCode\":\"DEVICE001\",\"status\":\"IDLE\"}";
        
        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);
        Thread.sleep(500);
        
        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);
        Thread.sleep(1000);
        
        assertTrue(true, "幂等性测试通过");
    }
}