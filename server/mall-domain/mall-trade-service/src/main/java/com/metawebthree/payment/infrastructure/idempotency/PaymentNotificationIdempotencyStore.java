package com.metawebthree.payment.infrastructure.idempotency;

import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.sql.Timestamp;
import java.time.LocalDateTime;

@Slf4j
@Component
@RequiredArgsConstructor
public class PaymentNotificationIdempotencyStore {
    private final JdbcTemplate jdbcTemplate;

    @PostConstruct
    public void init() {
        try {
            jdbcTemplate.execute(
                    "CREATE TABLE IF NOT EXISTS payment_notification (" +
                            "id BIGINT PRIMARY KEY AUTO_INCREMENT, " +
                            "channel VARCHAR(32) NOT NULL, " +
                            "external_tx_no VARCHAR(128) NOT NULL, " +
                            "order_no VARCHAR(64) NOT NULL, " +
                            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, " +
                            "UNIQUE (channel, external_tx_no))"
            );
        } catch (Exception e) {
            log.warn("Failed to initialize payment_notification table", e);
        }
    }

    public boolean tryRecord(String channel, String externalTxNo, String orderNo) {
        if (channel == null || channel.isBlank() || externalTxNo == null || externalTxNo.isBlank()) {
            return true;
        }
        try {
            jdbcTemplate.update(
                    "INSERT INTO payment_notification (channel, external_tx_no, order_no, created_at) VALUES (?, ?, ?, ?)",
                    channel, externalTxNo, orderNo, Timestamp.valueOf(LocalDateTime.now())
            );
            return true;
        } catch (DuplicateKeyException e) {
            log.debug("Duplicate payment notification: channel={}, txNo={}", channel, externalTxNo);
            return false;
        } catch (Exception e) {
            log.warn("Failed to record payment notification: channel={}, txNo={}, error={}", channel, externalTxNo, e.getMessage());
            return true;
        }
    }
}
