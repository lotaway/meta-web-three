package com.metawebthree.common.autoconfigure;

import org.apache.rocketmq.client.producer.DefaultMQProducer;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Import;

import com.metawebthree.common.utils.RocketMQ.MQConsumer;
import com.metawebthree.common.utils.RocketMQ.MQProducer;

/**
 * RocketMQ producer/consumer helpers, opt-in per service.
 */
@AutoConfiguration
@ConditionalOnClass(DefaultMQProducer.class)
@ConditionalOnProperty(prefix = "meta.common.storage.rocketmq", name = "enabled", havingValue = "true")
@EnableConfigurationProperties(CommonStorageProperties.class)
@Import({MQProducer.class, MQConsumer.class})
public class RocketMQAutoConfiguration {
}
