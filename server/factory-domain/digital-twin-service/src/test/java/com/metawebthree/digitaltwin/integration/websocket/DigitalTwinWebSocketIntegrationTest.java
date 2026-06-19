package com.metawebthree.digitaltwin.integration.websocket;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.client.WebSocketClient;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DisplayName("WebSocket 集成测试")
class DigitalTwinWebSocketIntegrationTest {

    private static final Logger log = LoggerFactory.getLogger(DigitalTwinWebSocketIntegrationTest.class);

    @LocalServerPort
    private int port;

    @Test
    @DisplayName("WebSocket 连接建立")
    void testWebSocketConnection() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicBoolean connected = new AtomicBoolean(false);

        WebSocketClient client = createClient();
        client.doHandshake(createConnectingHandler(latch, connected), getWsUrl()).get(5, TimeUnit.SECONDS);

        assertTrue(latch.await(5, TimeUnit.SECONDS), "连接超时");
        assertTrue(connected.get(), "连接未建立");
    }

    @Test
    @DisplayName("WebSocket 消息接收")
    void testWebSocketMessageReceiving() throws Exception {
        CountDownLatch messageLatch = new CountDownLatch(1);
        AtomicReference<String> receivedMessage = new AtomicReference<>();

        WebSocketClient client = createClient();
        client.doHandshake(createMessageHandler(messageLatch, receivedMessage), getWsUrl()).get(5, TimeUnit.SECONDS);

        assertTrue(messageLatch.await(5, TimeUnit.SECONDS), "消息接收超时");
        assertEquals("SUBSCRIBE", receivedMessage.get());
    }

    @Test
    @DisplayName("WebSocket 广播功能")
    void testWebSocketBroadcast() throws Exception {
        CountDownLatch latch = new CountDownLatch(2);

        WebSocketClient client1 = createClient();
        WebSocketClient client2 = createClient();

        client1.doHandshake(createSimpleHandler(latch), getWsUrl()).get(5, TimeUnit.SECONDS);
        client2.doHandshake(createSimpleHandler(latch), getWsUrl()).get(5, TimeUnit.SECONDS);

        assertTrue(latch.await(10, TimeUnit.SECONDS), "多个客户端连接超时");
    }

    @Test
    @DisplayName("WebSocket 连接关闭")
    void testWebSocketConnectionClose() throws Exception {
        CountDownLatch closeLatch = new CountDownLatch(1);

        WebSocketClient client = createClient();
        client.doHandshake(createClosingHandler(closeLatch), getWsUrl()).get(5, TimeUnit.SECONDS);

        assertTrue(closeLatch.await(5, TimeUnit.SECONDS), "连接关闭事件未收到");
    }

    private String getWsUrl() {
        return "ws://localhost:" + port + "/ws";
    }

    private WebSocketClient createClient() {
        return new StandardWebSocketClient();
    }

    private TextWebSocketHandler createConnectingHandler(CountDownLatch latch, AtomicBoolean connected) {
        return new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                connected.set(true);
                latch.countDown();
            }

            @Override
            protected void handleTextMessage(org.springframework.web.socket.WebSocketSession session,
                    org.springframework.web.socket.TextMessage message) {
            }
        };
    }

    private TextWebSocketHandler createMessageHandler(CountDownLatch latch, AtomicReference<String> receivedMessage) {
        return new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                try {
                    session.sendMessage(new org.springframework.web.socket.TextMessage(
                        "{\"type\":\"SUBSCRIBE\",\"workshopId\":\"WS001\"}"));
                } catch (Exception e) {
                    log.error("Failed to send message", e);
                    throw new RuntimeException("Failed to send message", e);
                }
            }

            @Override
            protected void handleTextMessage(org.springframework.web.socket.WebSocketSession session,
                    org.springframework.web.socket.TextMessage message) {
                receivedMessage.set(message.getPayload());
                latch.countDown();
            }
        };
    }

    private TextWebSocketHandler createSimpleHandler(CountDownLatch latch) {
        return new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                latch.countDown();
            }
        };
    }

    private TextWebSocketHandler createClosingHandler(CountDownLatch latch) {
        return new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                try {
                    session.close(org.springframework.web.socket.CloseStatus.NORMAL);
                } catch (Exception e) {
                    log.error("Failed to close connection", e);
                    throw new RuntimeException("Failed to close connection", e);
                }
            }

            @Override
            public void afterConnectionClosed(org.springframework.web.socket.WebSocketSession session,
                    org.springframework.web.socket.CloseStatus status) {
                latch.countDown();
            }
        };
    }
}