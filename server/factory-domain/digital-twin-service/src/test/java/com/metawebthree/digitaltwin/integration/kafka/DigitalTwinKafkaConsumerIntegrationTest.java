package com.metawebthree.digitaltwin.integration.kafka;

import com.metawebthree.digitaltwin.interfaces.websocket.DigitalTwinWebSocketHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.test.context.EmbeddedKafka;
import org.springframework.test.annotation.DirtiesContext;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@EmbeddedKafka(partitions = 1, topics = {"device.status.changed", "device.position.updated",
    "device.heartbeat", "alert.created", "production.output.updated", "agv.position.updated"},
    brokerProperties = {"listeners=PLAINTEXT://localhost:9092", "port=9092"})
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_EACH_TEST_METHOD)
@DisplayName("Kafka 消费集成测试")
class DigitalTwinKafkaConsumerIntegrationTest {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @Autowired(required = false)
    private DigitalTwinWebSocketHandler webSocketHandler;

    private final AtomicReference<String> lastMessage = new AtomicReference<>();

    @BeforeEach
    void setUp() {
        lastMessage.set(null);
    }

    private boolean waitForMessage(CountDownLatch latch, long timeoutMs) throws InterruptedException {
        return latch.await(timeoutMs, TimeUnit.MILLISECONDS);
    }

    @Test
    @DisplayName("消费 device.status.changed 消息")
    void consumeDeviceStatusChanged() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"messageId\":\"test-status-001\",\"deviceCode\":\"DEVICE001\",\"status\":\"RUNNING\"}";
        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
        assertNotNull(lastMessage.get(), "消息内容为空");
        assertTrue(lastMessage.get().contains("DEVICE001"), "设备编码不匹配");
    }

    @Test
    @DisplayName("消费 device.position.updated 消息")
    void consumeDevicePositionUpdated() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"messageId\":\"test-position-001\",\"deviceCode\":\"DEVICE001\",\"x\":100.5,\"y\":200.3}";
        kafkaTemplate.send("device.position.updated", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
        assertNotNull(lastMessage.get(), "消息内容为空");
    }

    @Test
    @DisplayName("消费 alert.created 消息")
    void consumeAlertCreated() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"messageId\":\"test-alert-001\",\"alertLevel\":\"HIGH\",\"content\":\"Temperature exceeded\"}";
        kafkaTemplate.send("alert.created", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
        assertNotNull(lastMessage.get(), "消息内容为空");
    }

    @Test
    @DisplayName("消费 production.output.updated 消息")
    void consumeProductionOutputUpdated() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"messageId\":\"test-output-001\",\"lineId\":\"LINE001\",\"output\":1500}";
        kafkaTemplate.send("production.output.updated", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
        assertNotNull(lastMessage.get(), "消息内容为空");
    }

    @Test
    @DisplayName("消费 agv.position.updated 消息")
    void consumeAgvPositionUpdated() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"messageId\":\"test-agv-001\",\"agvCode\":\"AGV001\",\"x\":50.0,\"y\":75.0,\"angle\":90}";
        kafkaTemplate.send("agv.position.updated", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
        assertNotNull(lastMessage.get(), "消息内容为空");
    }

    @Test
    @DisplayName("消费 device.heartbeat 消息")
    void consumeDeviceHeartbeat() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        String message = "{\"deviceCode\":\"DEVICE001\",\"timestamp\":1699999999000}";
        kafkaTemplate.send("device.heartbeat", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(latch, 3000), "消息未被消费者处理");
    }

    @Test
    @DisplayName("幂等性测试 - 重复消息只处理一次")
    void idempotencyTest() throws Exception {
        CountDownLatch firstLatch = new CountDownLatch(1);
        String messageId = "test-idempotent-001";
        String message = "{\"messageId\":\"" + messageId + "\",\"deviceCode\":\"DEVICE001\",\"status\":\"IDLE\"}";

        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);

        assertTrue(waitForMessage(firstLatch, 3000), "首次消息未被消费者处理");

        String firstMessageContent = lastMessage.get();
        assertNotNull(firstMessageContent, "消息内容为空");

        // 第二次发送同一消息，验证幂等性
        CountDownLatch secondLatch = new CountDownLatch(1);
        lastMessage.set(null);
        kafkaTemplate.send("device.status.changed", message).get(5, TimeUnit.SECONDS);

        // 使用确定的等待方式而不是 sleep
        boolean receivedSecond = waitForMessage(secondLatch, 1000);

        assertFalse(receivedSecond, "重复消息未被过滤，幂等性测试失败");
        assertNull(lastMessage.get(), "重复消息被处理了，幂等性测试失败");
    }
}