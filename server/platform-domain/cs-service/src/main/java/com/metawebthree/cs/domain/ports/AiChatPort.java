package com.metawebthree.cs.domain.ports;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

public interface AiChatPort {
    String chat(String sessionId, List<Map<String, String>> messages);
    String chatWithTools(String sessionId, List<Map<String, String>> messages,
                         List<Map<String, Object>> tools,
                         Function<Map<String, Object>, String> toolExecutor);
    boolean isAvailable();
}
