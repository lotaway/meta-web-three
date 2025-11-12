package com.metawebthree.media.setting;

import jakarta.websocket.*;
import jakarta.websocket.server.PathParam;
import jakarta.websocket.server.ServerEndpoint;
import lombok.extern.slf4j.Slf4j;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.media.DTO.DanmuMessageDTO;

import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
@ServerEndpoint("/ws/{username}")
public class WebSocketConfig {

    private static final ConcurrentHashMap<String, Session> sessions = new ConcurrentHashMap<>();
    private static final ConcurrentLinkedQueue<String> danmuHistory = new ConcurrentLinkedQueue<>();
    private static final int MAX_HISTORY = 100;
    private static final long RATE_LIMIT_MS = 1000;
    private static final ConcurrentHashMap<String, AtomicLong> lastSendTimes = new ConcurrentHashMap<>();
    private static final ObjectMapper objectMapper = new ObjectMapper();

    @OnOpen
    public void onOpen(Session session, @PathParam("username") String username) {
        sessions.put(username, session);
        sendSystemMessage(username + " 加入了房间");
        // Send history to new user
        danmuHistory.forEach(msg -> {
            try {
                session.getBasicRemote().sendText(msg);
            } catch (IOException e) {
                log.error("Failed to send history message", e);
            }
        });
    }

    @OnMessage
    public void onMessage(String message, Session session, @PathParam("username") String username) {
        // Rate limiting
        AtomicLong lastSendTime = lastSendTimes.computeIfAbsent(username, k -> new AtomicLong(0));
        long now = System.currentTimeMillis();
        if (now - lastSendTime.get() < RATE_LIMIT_MS) {
            return; // Rate limited
        }
        lastSendTime.set(now);

        try {
            DanmuMessageDTO danmu = objectMapper.readValue(message, DanmuMessageDTO.class);
            if (danmu.getNickname() == null) {
                // @TODO get nickname by username
                danmu.setNickname("Unknown User");
            }
            String formattedMsg = objectMapper.writeValueAsString(danmu);
            
            // Add to history and maintain size
            danmuHistory.add(formattedMsg);
            if (danmuHistory.size() > MAX_HISTORY) {
                danmuHistory.poll();
            }
            
            sendMessageToAll(formattedMsg);
        } catch (IOException e) {
            log.error("Failed to process danmu message", e);
        }
    }

    @OnClose
    public void onClose(Session session, @PathParam("username") String username) {
        sessions.remove(username);
        lastSendTimes.remove(username);
        sendSystemMessage(username + " 离开了房间");
    }

    @OnError
    public void onError(Session session, Throwable throwable) {
        log.error("WebSocket发生错误：" + throwable.getMessage());
    }

    private void sendMessageToAll(String message) {
        sessions.values().forEach(s -> {
            if (s.isOpen()) {
                try {
                    s.getBasicRemote().sendText(message);
                } catch (Exception e) {
                    log.error("Failed to send message", e);
                }
            }
        });
    }

    private void sendSystemMessage(String message) {
        try {
            String formattedMsg = objectMapper.writeValueAsString(new DanmuMessage() {{
                content = "系统提示: " + message;
                color = "#FF0000";
                position = "top";
                size = 16;
            }});
            sendMessageToAll(formattedMsg);
        } catch (IOException e) {
            log.error("Failed to format system message", e);
        }
    }
}
