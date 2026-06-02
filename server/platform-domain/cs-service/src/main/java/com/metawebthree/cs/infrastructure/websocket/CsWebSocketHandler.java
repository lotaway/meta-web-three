package com.metawebthree.cs.infrastructure.websocket;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.websocket.OnClose;
import jakarta.websocket.OnError;
import jakarta.websocket.OnMessage;
import jakarta.websocket.OnOpen;
import jakarta.websocket.Session;
import jakarta.websocket.server.PathParam;
import jakarta.websocket.server.ServerEndpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

@ServerEndpoint("/ws/cs/{sessionId}")
public class CsWebSocketHandler {
    private static final Logger log = LoggerFactory.getLogger(CsWebSocketHandler.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static SessionManager sessionManager;

    public static void setSessionManager(SessionManager sm) {
        sessionManager = sm;
    }

    @OnOpen
    public void onOpen(Session session, @PathParam("sessionId") String sessionId) {
        session.setMaxIdleTimeout(1800000);
    }

    @OnMessage
    public void onMessage(String message, Session session, @PathParam("sessionId") String sessionId) {
        try {
            JsonNode json = objectMapper.readTree(message);
            String type = json.get("type").asText();
            switch (type) {
                case "PING":
                    sendMessage(session, "{\"type\":\"PONG\"}");
                    break;
                case "REGISTER_AGENT":
                    handleAgentRegister(json, session);
                    break;
                case "REGISTER_CUSTOMER":
                    handleCustomerRegister(sessionId, session);
                    break;
                case "SEND_MESSAGE":
                    handleSendMessage(json, sessionId, session);
                    break;
                case "TYPING":
                    handleTyping(json, sessionId);
                    break;
                case "MARK_READ":
                    handleMarkRead(json, sessionId);
                    break;
                case "CLOSE_SESSION":
                    handleCloseSession(sessionId);
                    break;
                case "RATING":
                    handleRating(json, sessionId);
                    break;
                default:
                    sendMessage(session, "{\"type\":\"ERROR\",\"code\":\"UNKNOWN_TYPE\"}");
            }
        } catch (Exception e) {
            log.error("ws message error session:{}", sessionId, e);
            try {
                sendMessage(session, "{\"type\":\"ERROR\",\"code\":\"PARSE_ERROR\"}");
            } catch (IOException ex) {
            }
        }
    }

    private void handleAgentRegister(JsonNode json, Session session) {
        Long agentId = json.get("agentId").asLong();
        sessionManager.registerAgent(agentId, session);
    }

    private void handleCustomerRegister(String sessionId, Session session) {
        sessionManager.registerCustomer(sessionId, session);
    }

    private void handleSendMessage(JsonNode json, String sessionId, Session session) throws IOException {
        String targetSession = json.has("targetSession") ? json.get("targetSession").asText() : sessionId;
        String msg = objectMapper.writeValueAsString(
                java.util.Map.of("type", "NEW_MESSAGE",
                        "sessionId", targetSession,
                        "content", json.get("content").asText(),
                        "msgType", json.get("msgType").asText()));
        sessionManager.sendToCustomer(targetSession, msg);
        Long agentId = findAgentIdBySession(targetSession);
        if (agentId != null) {
            sessionManager.sendToAgent(agentId, msg);
        }
    }

    private void handleTyping(JsonNode json, String sessionId) {
        String targetSession = json.has("targetSession") ? json.get("targetSession").asText() : sessionId;
        String msg = "{\"type\":\"TYPING\",\"sessionId\":\"" + targetSession + "\"}";
        sessionManager.sendToCustomer(targetSession, msg);
    }

    private void handleMarkRead(JsonNode json, String sessionId) {
        String targetSession = json.has("targetSession") ? json.get("targetSession").asText() : sessionId;
        sessionManager.sendToCustomer(targetSession, "{\"type\":\"MARKED_READ\"}");
    }

    private void handleCloseSession(String sessionId) {
        sessionManager.sendToCustomer(sessionId, "{\"type\":\"SESSION_CLOSED\"}");
    }

    private void handleRating(JsonNode json, String sessionId) {
        String msg = "{\"type\":\"RATED\",\"score\":" + json.get("score") + "}";
        Long agentId = findAgentIdBySession(sessionId);
        if (agentId != null) {
            sessionManager.sendToAgent(agentId, msg);
        }
    }

    private Long findAgentIdBySession(String sessionId) {
        return sessionManager.findAgentIdBySessionId(sessionId);
    }

    private void sendMessage(Session session, String message) throws IOException {
        if (session.isOpen()) {
            session.getBasicRemote().sendText(message);
        }
    }

    @OnClose
    public void onClose(Session session, @PathParam("sessionId") String sessionId) {
        sessionManager.remove(session);
    }

    @OnError
    public void onError(Session session, Throwable throwable, @PathParam("sessionId") String sessionId) {
        log.error("ws error session:{}", sessionId, throwable);
        sessionManager.remove(session);
    }
}
