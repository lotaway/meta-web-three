package com.metawebthree.cs.domain.ports;

import java.util.List;
import java.util.Map;

public interface AiChatPort {
    String chat(String sessionId, List<Map<String, String>> messages);
    boolean isAvailable();
}
