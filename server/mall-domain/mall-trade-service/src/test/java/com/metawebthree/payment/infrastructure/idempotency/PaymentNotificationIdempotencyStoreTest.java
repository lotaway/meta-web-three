package com.metawebthree.payment.infrastructure.idempotency;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;

import javax.sql.DataSource;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PaymentNotificationIdempotencyStoreTest {

    private PaymentNotificationIdempotencyStore store;

    @BeforeEach
    void setUp() {
        String jdbcUrl = "jdbc:h2:mem:payment-idempotency-" + System.nanoTime()
                + ";MODE=MySQL;DB_CLOSE_DELAY=-1";
        DataSource dataSource = new DriverManagerDataSource(jdbcUrl, "sa", "");
        store = new PaymentNotificationIdempotencyStore(new JdbcTemplate(dataSource));
        store.init();
    }

    @Test
    void firstRecordReturnsTrue() {
        assertTrue(store.tryRecord("ALIPAY", "TX-1", "ORDER_1"));
    }

    @Test
    void duplicateRecordReturnsFalse() {
        assertTrue(store.tryRecord("ALIPAY", "TX-1", "ORDER_1"));
        assertFalse(store.tryRecord("ALIPAY", "TX-1", "ORDER_1"));
    }

    @Test
    void differentTransactionIsRecorded() {
        assertTrue(store.tryRecord("ALIPAY", "TX-1", "ORDER_1"));
        assertTrue(store.tryRecord("ALIPAY", "TX-2", "ORDER_1"));
    }

    @Test
    void differentChannelIsRecorded() {
        assertTrue(store.tryRecord("ALIPAY", "TX-1", "ORDER_1"));
        assertTrue(store.tryRecord("EXCHANGE", "TX-1", "ORDER_1"));
    }

    @Test
    void blankTransactionIsNotTreatedAsDuplicate() {
        assertTrue(store.tryRecord("ALIPAY", "", "ORDER_1"));
        assertTrue(store.tryRecord("ALIPAY", null, "ORDER_1"));
    }
}
