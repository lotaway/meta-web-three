package com.metawebthree.finance.domain.entity.arap;

import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import java.time.LocalDate;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for AccountsPayable entity
 */
class AccountsPayableTest {

    private void assertBigDecimalEquals(BigDecimal expected, BigDecimal actual) {
        assertEquals(0, expected.compareTo(actual));
    }

    @Test
    void testCreateAp() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        assertEquals("AP001", ap.getApCode());
        assertEquals(2001L, ap.getSupplierId());
        assertEquals("Supplier B", ap.getSupplierName());
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ap.getPaidAmount());
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getRemainingAmount());
        assertEquals(LocalDate.of(2026, 3, 2), ap.getDueDate());
        assertEquals(AccountsPayable.ApStatus.PENDING, ap.getStatus());
        assertTrue(ap.getIsActive());
    }

    @Test
    void testPayPartialAmount() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.pay(new BigDecimal("5000.00"));
        
        assertBigDecimalEquals(new BigDecimal("5000.00"), ap.getPaidAmount());
        assertBigDecimalEquals(new BigDecimal("15000.00"), ap.getRemainingAmount());
        assertEquals(AccountsPayable.ApStatus.PARTIAL_PAID, ap.getStatus());
    }

    @Test
    void testPayFullAmount() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.pay(new BigDecimal("20000.00"));
        
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getPaidAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ap.getRemainingAmount());
        assertEquals(AccountsPayable.ApStatus.PAID, ap.getStatus());
    }

    @Test
    void testPayExcessAmount() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.pay(new BigDecimal("30000.00"));
        
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getPaidAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ap.getRemainingAmount());
        assertEquals(AccountsPayable.ApStatus.PAID, ap.getStatus());
    }

    @Test
    void testPayZeroAmount() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.pay(BigDecimal.ZERO);
        
        assertBigDecimalEquals(BigDecimal.ZERO, ap.getPaidAmount());
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getRemainingAmount());
        assertEquals(AccountsPayable.ApStatus.PENDING, ap.getStatus());
    }

    @Test
    void testWriteOff() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.writeOff(new BigDecimal("8000.00"));
        
        assertBigDecimalEquals(new BigDecimal("8000.00"), ap.getPaidAmount());
        assertBigDecimalEquals(new BigDecimal("12000.00"), ap.getRemainingAmount());
    }

    @Test
    void testWriteOffFullAmount() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.writeOff(new BigDecimal("20000.00"));
        
        assertBigDecimalEquals(new BigDecimal("20000.00"), ap.getPaidAmount());
        assertBigDecimalEquals(BigDecimal.ZERO, ap.getRemainingAmount());
        assertEquals(AccountsPayable.ApStatus.WRITE_OFF, ap.getStatus());
    }

    @Test
    void testUpdateRelatedDocument() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.updateRelatedDocument("PO", "PO-2026-001");
        
        assertEquals("PO", ap.getRelatedDocumentType());
        assertEquals("PO-2026-001", ap.getRelatedDocumentNo());
    }

    @Test
    void testUpdateExchangeRate() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("10000.00"), LocalDate.of(2026, 1, 1),
                  60, "USD", 1L, "Admin");
        
        ap.updateExchangeRate(new BigDecimal("7.20"));
        
        assertBigDecimalEquals(new BigDecimal("72000.00"), ap.getAmount());
        assertBigDecimalEquals(new BigDecimal("7.20"), ap.getExchangeRate());
    }

    @Test
    void testDeactivate() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  60, "CNY", 1L, "Admin");
        
        ap.deactivate();
        
        assertFalse(ap.getIsActive());
    }

    @Test
    void testDefaultCreditTerm() {
        AccountsPayable ap = new AccountsPayable();
        ap.create("AP001", 2001L, "Supplier B", "PURCHASE", 
                  new BigDecimal("20000.00"), LocalDate.of(2026, 1, 1),
                  null, "CNY", 1L, "Admin");
        
        assertEquals(30, ap.getCreditTerm());
        assertEquals(LocalDate.of(2026, 1, 31), ap.getDueDate());
    }
}