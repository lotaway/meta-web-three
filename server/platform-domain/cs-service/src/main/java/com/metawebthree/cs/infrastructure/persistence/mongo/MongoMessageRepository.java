package com.metawebthree.cs.infrastructure.persistence.mongo;

import com.metawebthree.cs.domain.model.Message;
import com.metawebthree.cs.domain.repository.MessageRepository;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;

import java.util.List;

public class MongoMessageRepository implements MessageRepository {
    private final MongoTemplate mongoTemplate;

    public MongoMessageRepository(MongoTemplate mongoTemplate) {
        this.mongoTemplate = mongoTemplate;
    }

    @Override
    public Message save(Message message) {
        return mongoTemplate.save(message, "cs_message");
    }

    @Override
    public List<Message> findBySessionId(String sessionId) {
        Query query = Query.query(Criteria.where("sessionId").is(sessionId));
        return mongoTemplate.find(query, Message.class, "cs_message");
    }

    @Override
    public List<Message> findBySessionIdAfter(String sessionId, String afterMessageId) {
        Query query = Query.query(
                Criteria.where("sessionId").is(sessionId)
                        .and("id").gt(afterMessageId)
        );
        return mongoTemplate.find(query, Message.class, "cs_message");
    }
}
