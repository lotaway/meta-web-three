package com.metawebthree.cs.configuration;

import com.metawebthree.cs.application.AiRoutingService;
import com.metawebthree.cs.domain.ports.AiChatPort;
import com.metawebthree.cs.domain.repository.MessageRepository;
import com.metawebthree.cs.infrastructure.client.LocalLlmProviderClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class CsConfig {

    @Value("${spring.ai.openai.base-url:https://api.openai.com}")
    private String openaiBaseUrl;

    @Value("${spring.ai.openai.api-key:}")
    private String openaiApiKey;

    @Bean
    public AiChatPort aiChatPort() {
        return new LocalLlmProviderClient(openaiBaseUrl, openaiApiKey);
    }

    @Bean
    public AiRoutingService aiRoutingService(AiChatPort aiChatPort, MessageRepository messageRepository) {
        return new AiRoutingService(aiChatPort, messageRepository);
    }
}