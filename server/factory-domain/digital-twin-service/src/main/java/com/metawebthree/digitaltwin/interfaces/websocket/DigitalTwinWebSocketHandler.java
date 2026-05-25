package com.metawebthree.digitaltwin.interfaces.websocket;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.PingMessage;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import jakarta.annotation.PreDestroy;
import java.io.IOException;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public class DigitalTwinWebSocketHandler extends TextWebSocketHandler {

    private final Set<WebSocketSession> sessions = ConcurrentHashMap.newKeySet();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        String userId = (String) session.getAttributes().get("userId");
        String userRole = (String) session.getAttributes().get("userRole");

        sessions.add(session);
        log.info("WebSocket connected: sessionId={}, userId={}, role={}", session.getId(), userId, userRole);

        startHeartbeat(session);
    }

    private void startHeartbeat(WebSocketSession session) {
        scheduler.scheduleAtFixedRate(() -> {
            if (session.isOpen()) {
                try {
                    session.sendMessage(new PingMessage());
                } catch (IOException e) {
                    log.debug("Heartbeat failed for session {}, closing", session.getId());
                    try {
                        session.close(CloseStatus.GOING_AWAY);
                    } catch (IOException ex) {
                        log.error("Failed to close session", ex);
                    }
                }
            }
        }, 30, 30, TimeUnit.SECONDS);
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        sessions.remove(session);
        log.info("WebSocket disconnected: sessionId={}, status={}", session.getId(), status);
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) {
        log.error("WebSocket transport error: sessionId={}, error={}", session.getId(), exception.getMessage());
        sessions.remove(session);
        if (session.isOpen()) {
            try {
                session.close(CloseStatus.SERVER_ERROR);
            } catch (IOException e) {
                log.error("Failed to close session after error", e);
            }
        }
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) {
        String userId = (String) session.getAttributes().get("userId");
        log.debug("Received message from sessionId={}, userId={}: {}", session.getId(), userId, message.getPayload());
    }

    public void broadcast(Object message) {
        try {
            String json = objectMapper.writeValueAsString(message);
            TextMessage textMessage = new TextMessage(json);
            for (WebSocketSession session : sessions) {
                if (session.isOpen()) {
                    try {
                        session.sendMessage(textMessage);
                    } catch (IOException e) {
                        log.error("Failed to send message to session {}", session.getId(), e);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Failed to broadcast message", e);
        }
    }

    public void sendToSession(String sessionId, Object message) {
        try {
            String json = objectMapper.writeValueAsString(message);
            TextMessage textMessage = new TextMessage(json);
            for (WebSocketSession session : sessions) {
                if (session.getId().equals(sessionId) && session.isOpen()) {
                    session.sendMessage(textMessage);
                    break;
                }
            }
        } catch (Exception e) {
            log.error("Failed to send message to session {}", sessionId, e);
        }
    }

    public int getActiveSessionCount() {
        return (int) sessions.stream().filter(WebSocketSession::isOpen).count();
    }

    @PreDestroy
    public void shutdown() {
        log.info("Shutting down WebSocket handler, closing {} sessions", sessions.size());
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }

        for (WebSocketSession session : sessions) {
            if (session.isOpen()) {
                try {
                    session.close(CloseStatus.GOING_AWAY);
                } catch (IOException e) {
                    log.error("Failed to close session during shutdown", e);
                }
            }
        }
        sessions.clear();
    }
}