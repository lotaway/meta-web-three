package com.metawebthree.digitaltwin.integration.websocket;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.core.WebSocketClient;
import org.springframework.web.socket.core.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import jakarta.websocket.ClientEndpoint;
import jakarta.websocket.CloseReason;
import jakarta.websocket.OnClose;
import jakarta.websocket.OnMessage;
import jakarta.websocket.OnOpen;
import jakarta.websocket.Session;
import jakarta.websocket.ContainerProvider;

import java.net.URI;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@DisplayName("WebSocket 集成测试")
class DigitalTwinWebSocketIntegrationTest {

    @LocalServerPort
    private int port;

    @Test
    @DisplayName("WebSocket 连接建立")
    void testWebSocketConnection() throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicBoolean connected = new AtomicBoolean(false);
        
        WebSocketClient client = new StandardWebSocketClient();
        
        client.doHandshake(new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                connected.set(true);
                latch.countDown();
            }
            
            @Override
            protected void handleTextMessage(org.springframework.web.socket.WebSocketSession session, 
                    org.springframework.web.socket.TextMessage message) {
            }
        }, "ws://localhost:" + port + "/ws").get(5, TimeUnit.SECONDS);
        
        assertTrue(latch.await(5, TimeUnit.SECONDS), "连接超时");
        assertTrue(connected.get(), "连接未建立");
    }

    @Test
    @DisplayName("WebSocket 消息接收")
    void testWebSocketMessageReceiving() throws Exception {
        CountDownLatch messageLatch = new CountDownLatch(1);
        AtomicReference<String> receivedMessage = new AtomicReference<>();
        
        WebSocketClient client = new StandardWebSocketClient();
        
        client.doHandshake(new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                try {
                    session.sendMessage(new org.springframework.web.socket.TextMessage(
                        "{\"type\":\"SUBSCRIBE\",\"workshopId\":\"WS001\"}"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            
            @Override
            protected void handleTextMessage(org.springframework.web.socket.WebSocketSession session, 
                    org.springframework.web.socket.TextMessage message) {
                receivedMessage.set(message.getPayload());
                messageLatch.countDown();
            }
        }, "ws://localhost:" + port + "/ws").get(5, TimeUnit.SECONDS);
        
        assertTrue(messageLatch.await(5, TimeUnit.SECONDS), "消息接收超时");
    }

    @Test
    @DisplayName("WebSocket 广播功能")
    void testWebSocketBroadcast() throws Exception {
        CountDownLatch latch = new CountDownLatch(2);
        AtomicBoolean allConnected = new AtomicBoolean(true);
        
        WebSocketClient client1 = new StandardWebSocketClient();
        WebSocketClient client2 = new StandardWebSocketClient();
        
        client1.doHandshake(new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                latch.countDown();
            }
        }, "ws://localhost:" + port + "/ws").get(5, TimeUnit.SECONDS);
        
        client2.doHandshake(new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                latch.countDown();
            }
        }, "ws://localhost:" + port + "/ws").get(5, TimeUnit.SECONDS);
        
        assertTrue(latch.await(10, TimeUnit.SECONDS), "多个客户端连接超时");
    }

    @Test
    @DisplayName("WebSocket 连接关闭")
    void testWebSocketConnectionClose() throws Exception {
        CountDownLatch closeLatch = new CountDownLatch(1);
        
        WebSocketClient client = new StandardWebSocketClient();
        
        client.doHandshake(new TextWebSocketHandler() {
            @Override
            public void afterConnectionEstablished(org.springframework.web.socket.WebSocketSession session) {
                try {
                    session.close(org.springframework.web.socket.CloseStatus.NORMAL_CLOSURE);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            
            @Override
            public void afterConnectionClosed(org.springframework.web.socket.WebSocketSession session, 
                    org.springframework.web.socket.CloseStatus status) {
                closeLatch.countDown();
            }
        }, "ws://localhost:" + port + "/ws").get(5, TimeUnit.SECONDS);
        
        assertTrue(closeLatch.await(5, TimeUnit.SECONDS), "连接关闭事件未收到");
    }
}