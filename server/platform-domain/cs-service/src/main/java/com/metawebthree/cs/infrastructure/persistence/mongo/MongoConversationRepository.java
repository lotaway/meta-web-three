package com.metawebthree.cs.infrastructure.persistence.mongo;

import com.metawebthree.cs.domain.model.Conversation;
import com.metawebthree.cs.domain.model.enums.ConversationStatus;
import com.metawebthree.cs.domain.repository.ConversationRepository;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.mongodb.core.query.Update;

import java.util.List;
import java.util.Optional;

public class MongoConversationRepository implements ConversationRepository {
    private final MongoTemplate mongoTemplate;

    public MongoConversationRepository(MongoTemplate mongoTemplate) {
        this.mongoTemplate = mongoTemplate;
    }

    @Override
    public Conversation save(Conversation conversation) {
        return mongoTemplate.save(conversation, "cs_conversation");
    }

    @Override
    public Optional<Conversation> findBySessionId(String sessionId) {
        Query query = Query.query(Criteria.where("sessionId").is(sessionId));
        Conversation result = mongoTemplate.findOne(query, Conversation.class, "cs_conversation");
        return Optional.ofNullable(result);
    }

    @Override
    public Optional<Conversation> findActiveByCustomerId(Long customerId) {
        Query query = Query.query(
                Criteria.where("customerId").is(customerId)
                        .and("status").in(ConversationStatus.QUEUING, ConversationStatus.ACTIVE)
        );
        Conversation result = mongoTemplate.findOne(query, Conversation.class, "cs_conversation");
        return Optional.ofNullable(result);
    }

    @Override
    public List<Conversation> findByAgentId(Long agentId) {
        Query query = Query.query(Criteria.where("agentId").is(agentId));
        return mongoTemplate.find(query, Conversation.class, "cs_conversation");
    }

    @Override
    public List<Conversation> findByStatus(ConversationStatus status) {
        Query query = Query.query(Criteria.where("status").is(status));
        return mongoTemplate.find(query, Conversation.class, "cs_conversation");
    }

    @Override
    public List<Conversation> findByCustomerId(Long customerId) {
        Query query = Query.query(Criteria.where("customerId").is(customerId));
        return mongoTemplate.find(query, Conversation.class, "cs_conversation");
    }
}
