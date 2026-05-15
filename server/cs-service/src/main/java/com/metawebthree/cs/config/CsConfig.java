package com.metawebthree.cs.config;

import com.metawebthree.cs.domain.repository.AgentRepository;
import com.metawebthree.cs.domain.repository.ConversationRepository;
import com.metawebthree.cs.domain.repository.MessageRepository;
import com.metawebthree.cs.domain.repository.QuickReplyRepository;
import com.metawebthree.cs.application.AgentService;
import com.metawebthree.cs.application.AssignmentService;
import com.metawebthree.cs.application.ConversationService;
import com.metawebthree.cs.application.MessageService;
import com.metawebthree.cs.application.QuickReplyService;
import com.metawebthree.cs.domain.ports.ConversationEventPort;
import com.metawebthree.cs.infrastructure.persistence.mongo.MongoConversationRepository;
import com.metawebthree.cs.infrastructure.persistence.mongo.MongoMessageRepository;
import com.metawebthree.cs.infrastructure.persistence.mybatis.MybatisAgentMapper;
import com.metawebthree.cs.infrastructure.persistence.mybatis.MybatisAgentRepository;
import com.metawebthree.cs.infrastructure.persistence.mybatis.MybatisQuickReplyMapper;
import com.metawebthree.cs.infrastructure.persistence.mybatis.MybatisQuickReplyRepository;
import com.metawebthree.cs.infrastructure.websocket.CsWebSocketHandler;
import com.metawebthree.cs.infrastructure.websocket.SessionManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.mongodb.core.MongoTemplate;

@Configuration
public class CsConfig {
    private static final Logger log = LoggerFactory.getLogger(CsConfig.class);

    @Bean
    public SessionManager sessionManager() {
        SessionManager sm = new SessionManager();
        CsWebSocketHandler.setSessionManager(sm);
        return sm;
    }

    @Bean
    public ConversationRepository conversationRepository(MongoTemplate mongoTemplate) {
        return new MongoConversationRepository(mongoTemplate);
    }

    @Bean
    public MessageRepository messageRepository(MongoTemplate mongoTemplate) {
        return new MongoMessageRepository(mongoTemplate);
    }

    @Bean
    public AgentRepository agentRepository(MybatisAgentMapper mapper) {
        return new MybatisAgentRepository(mapper);
    }

    @Bean
    public QuickReplyRepository quickReplyRepository(MybatisQuickReplyMapper mapper) {
        return new MybatisQuickReplyRepository(mapper);
    }

    @Bean
    public ConversationEventPort conversationEventPort() {
        return (sessionId, event, detail) ->
                log.info("event session:{} type:{} detail:{}", sessionId, event, detail);
    }

    @Bean
    public ConversationService conversationService(
            ConversationRepository conversationRepository,
            ConversationEventPort conversationEventPort) {
        return new ConversationService(conversationRepository, conversationEventPort);
    }

    @Bean
    public MessageService messageService(MessageRepository messageRepository) {
        return new MessageService(messageRepository);
    }

    @Bean
    public AssignmentService assignmentService(
            AgentRepository agentRepository,
            ConversationRepository conversationRepository) {
        return new AssignmentService(agentRepository, conversationRepository);
    }

    @Bean
    public AgentService agentService(AgentRepository agentRepository) {
        return new AgentService(agentRepository);
    }

    @Bean
    public QuickReplyService quickReplyService(QuickReplyRepository quickReplyRepository) {
        return new QuickReplyService(quickReplyRepository);
    }
}
