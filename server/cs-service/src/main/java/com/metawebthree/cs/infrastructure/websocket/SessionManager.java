package com.metawebthree.cs.infrastructure.websocket;

import jakarta.websocket.Session;

import java.util.concurrent.ConcurrentHashMap;

public class SessionManager {
    private final ConcurrentHashMap<String, Session> agentConnections = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Session> customerConnections = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Session, String> sessionKeys = new ConcurrentHashMap<>();

    public void registerAgent(Long agentId, Session session) {
        String key = "agent:" + agentId;
        Session existing = agentConnections.put(key, session);
        if (existing != null && existing.isOpen()) {
            try { existing.close(); } catch (Exception e) { }
        }
        sessionKeys.put(session, key);
    }

    public void registerCustomer(String sessionId, Session session) {
        String key = "customer:" + sessionId;
        Session existing = customerConnections.put(key, session);
        if (existing != null && existing.isOpen()) {
            try { existing.close(); } catch (Exception e) { }
        }
        sessionKeys.put(session, key);
    }

    public void remove(Session session) {
        String key = sessionKeys.remove(session);
        if (key != null) {
            if (key.startsWith("agent:")) {
                agentConnections.remove(key);
            } else if (key.startsWith("customer:")) {
                customerConnections.remove(key);
            }
        }
    }

    public Session findAgentSession(Long agentId) {
        return agentConnections.get("agent:" + agentId);
    }

    public Session findCustomerSession(String sessionId) {
        return customerConnections.get("customer:" + sessionId);
    }

    public void sendToAgent(Long agentId, String message) {
        Session session = findAgentSession(agentId);
        if (session != null && session.isOpen()) {
            try {
                session.getBasicRemote().sendText(message);
            } catch (Exception e) {
            }
        }
    }

    public void sendToCustomer(String sessionId, String message) {
        Session session = findCustomerSession(sessionId);
        if (session != null && session.isOpen()) {
            try {
                session.getBasicRemote().sendText(message);
            } catch (Exception e) {
            }
        }
    }

    public int onlineAgentCount() {
        return agentConnections.size();
    }
}
