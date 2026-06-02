package com.metawebthree.finance.domain.entity.arap;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.LocalDate;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AccountsReceivable entity
 */
class AccountsReceivableTest {

    private void assertBigDecimalEquals(BigDecimal expected, BigDecimal actual) {
        assertEquals(0, expected.compareTo(actual));
    }

    @Test
    void testCreateAr() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        assertEquals("AR001", ar.getArCode());
        assertEquals(1001L, ar.getCustomerId());
        assertEquals("Customer A", ar.getCustomerName());
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ar.getReceivedAmount());
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getRemainingAmount());
        assertEquals(LocalDate.of(2026, 1, 31), ar.getDueDate());
        assertEquals(AccountsReceivable.ArStatus.PENDING, ar.getStatus());
        assertTrue(ar.getIsActive());
    }

    @Test
    void testReceivePartialAmount() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.receive(new BigDecimal("3000.00"));
        
        assertBigDecimalEquals(new BigDecimal("3000.00"), ar.getReceivedAmount());
        assertBigDecimalEquals(new BigDecimal("7000.00"), ar.getRemainingAmount());
        assertEquals(AccountsReceivable.ArStatus.PARTIAL_RECEIVED, ar.getStatus());
    }

    @Test
    void testReceiveFullAmount() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.receive(new BigDecimal("10000.00"));
        
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getReceivedAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ar.getRemainingAmount());
        assertEquals(AccountsReceivable.ArStatus.RECEIVED, ar.getStatus());
    }

    @Test
    void testReceiveExcessAmount() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.receive(new BigDecimal("15000.00"));
        
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getReceivedAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ar.getRemainingAmount());
        assertEquals(AccountsReceivable.ArStatus.RECEIVED, ar.getStatus());
    }

    @Test
    void testReceiveZeroAmount() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.receive(BigDecimal.ZERO);
        
        assertBigDecimalEquals(BigDecimal.ZERO, ar.getReceivedAmount());
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getRemainingAmount());
        assertEquals(AccountsReceivable.ArStatus.PENDING, ar.getStatus());
    }

    @Test
    void testWriteOff() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.writeOff(new BigDecimal("5000.00"));
        
        assertBigDecimalEquals(new BigDecimal("5000.00"), ar.getReceivedAmount());
        assertBigDecimalEquals(new BigDecimal("5000.00"), ar.getRemainingAmount());
    }

    @Test
    void testWriteOffFullAmount() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.writeOff(new BigDecimal("10000.00"));
        
        assertBigDecimalEquals(new BigDecimal("10000.00"), ar.getReceivedAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ar.getRemainingAmount());
        assertEquals(AccountsReceivable.ArStatus.WRITE_OFF, ar.getStatus());
    }

    @Test
    void testUpdateRelatedDocument() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.updateRelatedDocument("ORDER", "ORD-2026-001");
        
        assertEquals("ORDER", ar.getRelatedDocumentType());
        assertEquals("ORD-2026-001", ar.getRelatedDocumentNo());
    }

    @Test
    void testUpdateExchangeRate() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "USD", 1L, "Admin");
        
        ar.updateExchangeRate(new BigDecimal("7.20"));
        
        assertBigDecimalEquals(new BigDecimal("72000.00"), ar.getAmount());
        assertBigDecimalEquals(new BigDecimal("7.20"), ar.getExchangeRate());
    }

    @Test
    void testDeactivate() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  30, "CNY", 1L, "Admin");
        
        ar.deactivate();
        
        assertFalse(ar.getIsActive());
    }

    @Test
    void testDefaultCreditTerm() {
        AccountsReceivable ar = new AccountsReceivable();
        ar.create("AR001", 1001L, "Customer A", "SALE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  null, "CNY", 1L, "Admin");
        
        assertEquals(30, ar.getCreditTerm());
        assertEquals(LocalDate.of(2026, 1, 31), ar.getDueDate());
    }
}