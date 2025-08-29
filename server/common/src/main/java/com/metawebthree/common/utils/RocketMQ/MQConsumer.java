package com.metawebthree.common.utils.RocketMQ;

import org.apache.rocketmq.client.consumer.DefaultMQPushConsumer;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyContext;
import org.apache.rocketmq.client.consumer.listener.ConsumeConcurrentlyStatus;
import org.apache.rocketmq.client.consumer.listener.MessageListener;
import org.apache.rocketmq.client.consumer.listener.MessageListenerConcurrently;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.common.message.MessageExt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;

@Slf4j
@Configuration
public class MQConsumer {
    private final DefaultMQPushConsumer consumer;
    
    @Value("${rocketmq.client.namesrv:未配置}")
    private String namesrv;
    
    @Value("${rocketmq.client.appliaction-topic:未配置}")
    private String applicationTopic;
    
    public MQConsumer() {
        consumer = new DefaultMQPushConsumer("MQConsumer");
    }

    @PostConstruct
    public void init() {
        consumer.setNamesrvAddr(namesrv);
        log.info("RocketMQ Nameserver: " + namesrv);
        log.info("RocketMQ Common application config: " + applicationTopic);
    }

    public void start(String topic, MessageListener messageListener, @Nullable String subExpression) throws MQClientException {
        if (subExpression == null) {
            subExpression = "*";
        }
        consumer.subscribe(topic, subExpression);
        consumer.setMessageListener(messageListener);
        consumer.start();
    }

    public void test() throws MQClientException {
        start("TestTopic", new MessageListenerConcurrently() {
            @Override
            public ConsumeConcurrentlyStatus consumeMessage(List<MessageExt> list, ConsumeConcurrentlyContext consumeConcurrentlyContext) {
                for (int i = 0; i < list.size(); i++) {
                    MessageExt msg = list.get(i);
                    System.out.printf("Message: %d, %s", i, Arrays.toString(msg.getBody()));
                }
                return ConsumeConcurrentlyStatus.CONSUME_SUCCESS;
            }
        }, null);
    }

    public void end() {
        consumer.shutdown();
    }
}
