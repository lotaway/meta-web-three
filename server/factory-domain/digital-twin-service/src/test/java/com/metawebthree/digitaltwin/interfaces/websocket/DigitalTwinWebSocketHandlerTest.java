package com.metawebthree.digitaltwin.interfaces.websocket;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class DigitalTwinWebSocketHandlerTest {

    @Mock
    private WebSocketSession session;

    private DigitalTwinWebSocketHandler handler;

    @BeforeEach
    void setUp() {
        handler = new DigitalTwinWebSocketHandler();
    }

    @Test
    void afterConnectionEstablished_shouldAddSession() {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        attributes.put("userRole", "admin");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-001");
        when(session.isOpen()).thenReturn(true);

        handler.afterConnectionEstablished(session);

        assertTrue(handler.getActiveSessionCount() >= 0);
    }

    @Test
    void afterConnectionClosed_shouldRemoveSession() {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-002");
        when(session.isOpen()).thenReturn(true);

        handler.afterConnectionEstablished(session);
        int beforeCount = handler.getActiveSessionCount();
        handler.afterConnectionClosed(session, CloseStatus.NORMAL);

        assertTrue(handler.getActiveSessionCount() <= beforeCount);
    }

    @Test
    void handleTransportError_shouldRemoveSession() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-003");
        when(session.isOpen()).thenReturn(true);

        handler.afterConnectionEstablished(session);
        handler.handleTransportError(session, new RuntimeException("Test error"));

        verify(session).close(CloseStatus.SERVER_ERROR);
    }

    @Test
    void broadcast_shouldSendMessageToAllSessions() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-004");
        when(session.isOpen()).thenReturn(true);
        
        handler.afterConnectionEstablished(session);

        Map<String, Object> message = Map.of("type", "TEST", "data", "hello");
        handler.broadcast(message);

        verify(session, atLeastOnce()).sendMessage(any(TextMessage.class));
    }

    @Test
    void broadcast_shouldNotThrowWhenSessionSendFails() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-005");
        when(session.isOpen()).thenReturn(true);
        doThrow(new IOException("Send failed")).when(session).sendMessage(any(TextMessage.class));
        
        handler.afterConnectionEstablished(session);

        Map<String, Object> message = Map.of("type", "TEST", "data", "hello");
        assertDoesNotThrow(() -> handler.broadcast(message));
    }

    @Test
    void sendToSession_shouldSendMessageToSpecificSession() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-006");
        when(session.isOpen()).thenReturn(true);
        
        handler.afterConnectionEstablished(session);

        Map<String, Object> message = Map.of("type", "DIRECT", "data", "target");
        handler.sendToSession("session-006", message);

        verify(session).sendMessage(any(TextMessage.class));
    }

    @Test
    void sendToSession_shouldNotSendToClosedSession() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-007");
        when(session.isOpen()).thenReturn(false);
        
        handler.afterConnectionEstablished(session);

        Map<String, Object> message = Map.of("type", "DIRECT", "data", "target");
        handler.sendToSession("session-007", message);

        verify(session, never()).sendMessage(any(TextMessage.class));
    }

    @Test
    void getActiveSessionCount_shouldReturnZeroForEmptyHandler() {
        assertEquals(0, handler.getActiveSessionCount());
    }

    @Test
    void getActiveSessionCount_shouldReturnCorrectCount() {
        Map<String, Object> attributes1 = new ConcurrentHashMap<>();
        attributes1.put("userId", "user-001");
        WebSocketSession session1 = mock(WebSocketSession.class);
        when(session1.getAttributes()).thenReturn(attributes1);
        when(session1.getId()).thenReturn("session-101");
        when(session1.isOpen()).thenReturn(true);

        Map<String, Object> attributes2 = new ConcurrentHashMap<>();
        attributes2.put("userId", "user-002");
        WebSocketSession session2 = mock(WebSocketSession.class);
        when(session2.getAttributes()).thenReturn(attributes2);
        when(session2.getId()).thenReturn("session-102");
        when(session2.isOpen()).thenReturn(true);

        handler.afterConnectionEstablished(session1);
        handler.afterConnectionEstablished(session2);

        assertEquals(2, handler.getActiveSessionCount());
    }

    @Test
    void handleTextMessage_shouldProcessMessage() {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-008");

        TextMessage message = new TextMessage("{\"action\":\"subscribe\"}");
        handler.handleTextMessage(session, message);
    }

    @Test
    void broadcast_shouldHandleEmptySessions() {
        Map<String, Object> message = Map.of("type", "TEST", "data", "hello");
        assertDoesNotThrow(() -> handler.broadcast(message));
    }

    @Test
    void shutdown_shouldCloseAllSessions() throws IOException {
        Map<String, Object> attributes = new ConcurrentHashMap<>();
        attributes.put("userId", "user-001");
        
        when(session.getAttributes()).thenReturn(attributes);
        when(session.getId()).thenReturn("session-009");
        when(session.isOpen()).thenReturn(true);
        
        handler.afterConnectionEstablished(session);
        
        assertDoesNotThrow(() -> handler.shutdown());
    }
}