package com.metawebthree.mes.infrastructure.mqtt;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.paho.client.mqttv3.*;
import org.eclipse.paho.client.mqttv3.persist.MemoryPersistence;

@Slf4j
public class MqttCommandPublisher {

    private final String brokerUrl;
    private final String clientId;
    private final String topicPrefix;
    private final boolean enabled;
    private MqttClient client;

    public MqttCommandPublisher(String brokerUrl, String clientId, String topicPrefix, boolean enabled) {
        this.brokerUrl = brokerUrl;
        this.clientId = clientId;
        this.topicPrefix = topicPrefix;
        this.enabled = enabled;
    }

    public void connect() {
        if (!enabled) return;
        try {
            client = new MqttClient(brokerUrl, clientId, new MemoryPersistence());
            MqttConnectOptions opts = new MqttConnectOptions();
            opts.setCleanSession(true);
            opts.setAutomaticReconnect(true);
            client.connect(opts);
            log.info("MQTT publisher connected to {}", brokerUrl);
        } catch (MqttException e) {
            log.error("MQTT publisher connection failed: {}", e.getMessage());
        }
    }

    public boolean publish(String equipmentCode, String payload) {
        if (!enabled || client == null || !client.isConnected()) {
            log.warn("MQTT not available, command not sent to {}", equipmentCode);
            return false;
        }
        try {
            String topic = topicPrefix + equipmentCode + "/command";
            MqttMessage message = new MqttMessage(payload.getBytes());
            message.setQos(1);
            message.setRetained(false);
            client.publish(topic, message);
            log.info("MQTT published to {}: {}", topic, payload);
            return true;
        } catch (MqttException e) {
            log.error("MQTT publish failed for {}: {}", equipmentCode, e.getMessage());
            return false;
        }
    }

    public void disconnect() {
        if (client != null && client.isConnected()) {
            try {
                client.disconnect();
                client.close();
            } catch (MqttException e) {
                log.warn("MQTT disconnect error: {}", e.getMessage());
            }
        }
    }
}
